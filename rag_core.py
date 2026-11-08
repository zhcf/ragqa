"""RAG 核心：PDF 解析、切块、向量检索、答案生成、页码定位

文本提取从 PyPDF2 换成了 PyMuPDF，因为高亮要用它在同一份数据上给出的坐标，
两处文本同源才能保证检索到的片段在页面上找得到。
"""
import os
import re
import time
from collections import Counter
from pathlib import Path

import pymupdf
import torch
# PyInstaller 看不到 AutoModel 里按 model_type 字符串做的映射，显式导入一次，
# 打包后 bge-small-zh 才能找到它需要的 BertModel
import transformers.models.bert  # noqa: F401
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from config import (CHUNK_OVERLAP, CHUNK_SIZE, EMBED_PATH, MAX_ARTICLE_LEN,
                    MAX_NEW_TOKENS, PDF_PATH, QUERY_INSTRUCTION, TOP_K)

torch.set_num_threads(min(8, os.cpu_count() or 8))

NOT_FOUND = "资料中没有找到相关内容"
# 兜底句的动词短语：模型很少原样输出上面那句，常写成“资料中没有找到
# 关于 X 的相关内容”甚至“文中未提及”，统一靠动词短语识别
NOT_FOUND_HINT = re.compile(r"(没有|未|并未|无法|没能)\s*(找到|提及|提到|涉及|出现|包含|记载)")
SYSTEM_PROMPT = ("你是文档问答助手。只根据用户提供的资料回答，"
                 f"资料里没有的内容就说“{NOT_FOUND}”，不要编造。")

# 目录页特征：大量 "......" 引导点（PDF 抽出来常是 ". . . . . ."）
TOC_DOTS = re.compile(r"\.(?:\s*\.){2,}")
# 目录条目：满屏“第一章 / 第二节 xxx”，几乎没有完整句子
TOC_TITLE = re.compile(r"第\s*[一二三四五六七八九十百零\d]+\s*[章节]")
# 条文起点：句号/分号/换行后紧跟的“第 X 条”；正文里“依照本法第五十二条”不算
ARTICLE_START = re.compile(r"(?:(?<=[。；！？\n])|^)\s*(第[一二三四五六七八九十百零\d]+条)")
# 条号本身，用于把高亮从一块扩展到整条
ARTICLE_NUMBER = re.compile(r"第[一二三四五六七八九十百零\d]+条")
# 目录行：整行只有“第 X 章 / 第 X 节 + 标题”，没有标点
TOC_LINE = re.compile(r"^第\s*[一二三四五六七八九十百零\d]+\s*[章节]\S*$")
# 页码行：“15” “- 15 -” “第 15 页” “Page 15 of 100” “15 / 76”；
# 只在页面上下边缘删，正文里的年份、编号不动。罗马数字页码也认，但按标准
# 写法校验——“mix”“civil”这种单词恰好全是 ivxlcdm 字母，不能当页码删掉
PAGE_NUMBER_LINE = re.compile(
    r"^[\s\-—–－]*(?:page\s*)?(?:第\s*)?(?:\d{1,4}|"
    r"(?=[ivxlcdm])(?:m{0,3})(?:cm|cd|d?c{0,3})"
    r"(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3}))"
    r"(?:\s*(?:页|/\s*\d{1,4}|of\s+\d{1,4}))?[\s\-—–－]*$",
    re.I,
)
# 导航句：“有关其他参数的说明，请参阅第 59 页” 这类句子在向量上容易被误检索
NAV_SENTENCE = re.compile(r"请参阅|参见|详见|见第\s*\d+\s*页|请参考")
# 答案里出现的页码标记
PAGE_MARK = re.compile(r"第\s*\d+\s*页")
# 小模型会把 prompt 里的占位符原样吐出来，比如“第 N 页”
PLACEHOLDER = re.compile(r"[（(]?\s*第\s*[NnXx]\s*页\s*[）)]?")

_embed_model = None


def get_embed_model():
    """向量模型全局只加载一次"""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(str(EMBED_PATH))
    return _embed_model


# ====================== 切块 ======================
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """中文 PDF 抽出的文本几乎没有空格，按字符长度 + 句末标点切块"""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?:\.\s?){3,}", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)

    chunks, buf = [], ""
    for sent in re.split(r"(?<=[。！？；\n])", text):
        sent = sent.strip()
        if not sent:
            continue
        if len(buf) + len(sent) > chunk_size and buf:
            chunks.append(buf)
            buf = buf[-overlap:] if overlap else ""
        buf += sent
    if buf:
        chunks.append(buf)
    return chunks


def strip_page_number(text, page_no):
    """去掉开头混进正文的页码

    PDF 抽文字时页码常和第一行正文连在一起（"23第九十条…"），
    带着它切块，块首那句会以数字开头；高亮以它开头时，定位会跑到
    页面顶部或底部的页码上。只处理开头正好等于本页页码的情况。
    """
    body = text.lstrip()
    label = str(page_no)
    if body.startswith(label):
        return body[len(label):]
    return body


def drop_toc_lines(text):
    """删掉目录行：整行只有“第 X 章 / 第 X 节 + 标题”、没有标点

    这类行没有信息量，留在正文里会让“高速公路”这种目录词命中整个块，
    高亮只能标在页顶的目录上。正文句子都带标点，不受影响。
    """
    kept = []
    for line in text.splitlines():
        bare = line.strip()
        if (bare and len(bare) < 40 and TOC_LINE.match(bare)
                and not re.search(r"[。；，、]", bare)):
            continue
        kept.append(line)
    return "\n".join(kept)


# ====================== 版面清洗 ======================
def read_pages(doc, progress=None):
    """把每页拆成带字号的行，[{页码, 页高, [(文本, 字号, y0, y1, x0, x1, 表格行)]}]

    切块要用字号判断标题（字号比编号格式稳得多，不管标题写“4 查看队列
    中的作业信息”还是“第 4 章 …”都认得出），所以这里取 dict 而不是 text；
    高亮仍用 rawdict 现取坐标，两处文字同源。

    同一水平线上有三块以上、横向隔开的文本就是表格的一行（单元格），标上
    标记供标题判定排除——表格数据行（“76 GPU-V100 gpu.slurm”）以数字
    开头，看文字会被当成编号标题。按位置判断不依赖文档内容。
    """
    pages = []
    for page_no in range(doc.page_count):
        page = doc[page_no]
        rows = []
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(span.get("text", "") for span in spans).strip()
                if not text:
                    continue
                size = max((span.get("size", 0.0) for span in spans), default=0.0)
                x0, y0, x1, y1 = line["bbox"]
                rows.append((text, size, y0, y1, x0, x1))
        if progress is not None:
            progress(page_no + 1, doc.page_count)
        if not rows:
            continue

        # 找表格行只用于打标记，不能改输出顺序：get_text 的块顺序就是阅读
        # 顺序，按 y 全局重排会把两栏和脚注区域的行串到一起。同一行几个
        # 单元格是不同 block，靠 y 才能聚到一起；两栏排版同一水平也只有
        # 左右两块，三个以上才算表格，避免误伤
        order = sorted(range(len(rows)), key=lambda k: (round(rows[k][2], 1),
                                                       rows[k][4]))
        table_at, i = set(), 0
        while i < len(order):
            j = i + 1
            while (j < len(order)
                   and rows[order[j]][2] - rows[order[i]][2] <= rows[order[i]][1]):
                j += 1
            band = [rows[k] for k in order[i:j]]
            size = max(row[1] for row in band)
            gaps = [band[k + 1][4] - band[k][5] for k in range(len(band) - 1)]
            if len(band) >= 3 and any(gap > size * 2 for gap in gaps):
                table_at.update(order[i:j])
            i = j

        lines = [row + (row_no in table_at,) for row_no, row in enumerate(rows)]
        pages.append({"no": page_no + 1, "height": page.rect.height,
                      "lines": lines})
    return pages


def _head_sig(text):
    """页眉签名：数字（含全角）归一化后去空白

    “第 15 页”“Page 15 of 100”“4 查看队列中的作业信息：SQUEUE 15”每页只有
    数字不同，归一化之后才能跨页对上；章节号也跟着变成 #，不影响判断。
    """
    return re.sub(r"[0-9０-９]+", "#", re.sub(r"\s+", "", text))


def drop_page_furniture(pages, top=0.15, bottom=0.9, repeat=3):
    """删掉页眉页脚：贴在页面上下边缘、跨页重复的短行和页码行

    页眉页脚和正文的差别不在文字，在位置和重复：每页顶上的章节名、页码，
    每页底下的“第 X 页 共 Y 页”、版权行，都只在数字上不同。同一签名在
    至少三页的同一高度（相差不到 10 磅）出现才删，正文里偶然重复的步骤名
    不在边缘区域，不会误删；法规这种每页开头都是不同条文的文档，一页都
    删不到，原样返回。
    """
    counter = Counter()
    for page in pages:
        height = page["height"] or 1.0
        for text, _, y0, y1, *_ in page["lines"]:
            if len(text) <= 60 and (y0 < height * top or y1 > height * bottom):
                counter[(_head_sig(text), int(y0 // 10))] += 1
    repeated = {key for key, count in counter.items() if count >= repeat}

    cleaned = []
    for page in pages:
        height = page["height"] or 1.0
        kept = []
        for text, size, y0, y1, x0, x1, table in page["lines"]:
            edge = y0 < height * top or y1 > height * bottom
            if edge and (PAGE_NUMBER_LINE.match(text)
                         or (_head_sig(text), int(y0 // 10)) in repeated):
                continue
            kept.append((text, size, y0, y1, x0, x1, table))
        if kept:
            cleaned.append({**page, "lines": kept})
    return cleaned


def drop_toc_pages(pages):
    """丢掉整页都是目录的页：满屏 “....” 点线，没有正文"""
    kept = []
    for page in pages:
        text = "\n".join(line[0] for line in page["lines"])
        if len(TOC_DOTS.findall(text)) <= 5:
            kept.append(page)
    return kept


def body_font_size(pages):
    """正文字号：按字符数加权的众数字号（标题字数少，压不过正文）"""
    counter = Counter()
    for page in pages:
        for text, size, *_ in page["lines"]:
            counter[round(size, 1)] += len(text)
    return counter.most_common(1)[0][0] if counter else 0.0


# 通用标题编号：1 / 1.2 / 1. / 一、/（一）/ 第 4 章 / 第 3 节 / Chapter 4。
# 只认编号形式，不认具体文字：任何编号体系的手册都适用
HEAD_NUMBER = re.compile(
    r"^(?:"
    r"\d{1,2}(?:\.\d{1,3})+(?=[\s、:：]|$)"
    r"|\d{1,2}[.、](?=\s|$)"
    r"|\d{1,2}\s+"
    r"|第\s*[一二三四五六七八九十百零〇\d]+\s*[章节篇部]"
    r"|[一二三四五六七八九十]{1,3}[、.]"
    r"|[（(][一二三四五六七八九十\d]{1,3}[）)]"
    r"|chapter\s+\d+|appendix\b"
    r")",
    re.I,
)
HEAD_MAX_LEN = 60


def is_heading(text, size, body_size, table=False):
    """一行是不是标题：短、不带句末标点，且字号明显大于正文或带通用编号

    字号是比文字格式更稳的信号：手册正文 12 号、标题 17.2 号，标题写
    “4 查看队列中的作业信息”还是“第 4 章 …”都认得出。参数表条目的字号
    也偏大，但都以“。”结尾，靠标点排除；字号全线一样的文档退到编号判断，
    这时还要像标题——表格单元格（table），“2 rnode[02-03]”这种带括号的
    数据行都不是。
    """
    if table or not text or len(text) > HEAD_MAX_LEN or re.search(r"[。；]", text):
        return False
    if text[-1] in "，,、：:":
        return False
    if body_size:
        if size >= body_size * 1.25:
            return True
        if size >= body_size * 1.12 and HEAD_NUMBER.match(text):
            return True
    if len(text) > 40 or not HEAD_NUMBER.match(text):
        return False
    # 数据行带 [] {} = 这类代码符号；标题基本不带。中英文文档一样适用
    if re.search(r"[\[\]{}<>=|\\]", text):
        return False
    return sum(1 for ch in text if ch.isalpha()) >= len(text) * 0.3


def page_headings(lines, body_size):
    """找页内标题，返回 [(标题文本, 标题后第一行下标)]

    PDF 抽文字常把“4”和标题拆成两行、位置只差两三个点，同一基线（y 相差
    不超过字号）且首尾相接的相邻行合回“4 查看队列中的作业信息：squeue”；
    合并只看位置，不要求下一行也像标题——“主要输出项”字号只大一点点，
    这正是要合进来的那半句。横向隔着大片空白的是表格格子，不能合。
    标题单独成块后，“某功能用什么命令”这类问题不必再从 400 字的参数表里
    碰运气。
    """
    groups = []
    i = 0
    while i < len(lines):
        text, size, y0, _, x0, x1, table = lines[i]
        if not is_heading(text, size, body_size, table):
            i += 1
            continue
        parts, j, right, prev_y = [text], i + 1, x1, y0
        while (j < len(lines) and len(" ".join(parts)) < HEAD_MAX_LEN
               and abs(lines[j][2] - prev_y) <= size
               and lines[j][4] - right <= size * 2
               and not lines[j][6]):
            parts.append(lines[j][0])
            prev_y, right = lines[j][2], lines[j][5]
            j += 1
        groups.append((" ".join(parts), j))
        i = j
    return groups


def heading_lead(lines, start, body_size, limit=120):
    """标题后面紧跟的正文，取到第一句句号；紧挨着的下一个标题跳过去"""
    buf = ""
    for text, size, _, _, _, _, table in lines[start:]:
        if is_heading(text, size, body_size, table):
            continue
        buf += text
        if len(buf) >= limit:
            break
    for match in re.finditer(r"[。！？；]", buf):
        if match.end() >= 10:
            return buf[:match.end()]
    return buf[:limit]


def is_noise(chunk):
    if len(chunk) < 50 or chunk.count(".") / max(len(chunk), 1) > 0.3:
        return True
    # 目录条目块没有信息量：检索命中它，高亮只会标到一串章节名上
    return len(TOC_TITLE.findall(chunk)) >= 4 and chunk.count("。") <= 1


def split_articles(pages):
    """法规类文档按“第 X 条”切块，返回 [(条文, 起始页)]；条文太少返回 None

    400 字块会把几条混在一起、把一款从中间切断：问“事故后怎么做”时命中的
    块里可能只有“未造成人身伤亡”那一款，答案自然缺掉“停车、抢救、报警”
    那段；问“多久检验一次”则完全找不到第十三条。按条切让每条单独成一个
    检索单元，模型和标黄拿到的都是完整条文。
    条文拼页面时保留页码到位置的映射，跨页的条归到起始页。
    """
    full, page_at, offset = "", [], 0
    for page_no, text in pages:
        page_at.append((offset, page_no))
        full += text + "\n"
        offset = len(full)

    starts = [m.start(1) for m in ARTICLE_START.finditer(full)]
    if len(starts) < 5:                      # 零星几个“第 X 条”不算法规
        return None

    def page_of(pos):
        page = page_at[0][1]
        for begin, page_no in page_at:
            if begin <= pos:
                page = page_no
        return page

    result, bounds = [], starts + [len(full)]
    for i in range(len(starts)):
        body = full[bounds[i]:bounds[i + 1]].strip()
        if len(body) < 10:
            continue
        if len(body) <= MAX_ARTICLE_LEN:
            result.append((body, page_of(bounds[i])))
        else:
            # 超长条（附则、处罚条款常见）再切：按句切、不带重叠，切出来的
            # 每块都是完整句子。带重叠切会让下一块从上一块半句尾巴开始，
            # 检索命中它时高亮也从半句开始，对不上页面上的句子开头
            for piece in split_text(body, overlap=0):
                result.append((piece, page_of(bounds[i])))
    return result


# 标题块的分数折扣：标题块短，向量分天然偏高，问题里的词碰巧重合就会冲高
# （“如何限制作业的内存”曾被“11.4 OpenMP共享内存并行作业提交”抢走），
# 打折后只有明显更相关时才能压过正文块
HEAD_PENALTY = 0.92
# 分数几乎持平时（bge-small 对“检验/审验”这种一字之差只差万分之几）按
# 字面覆盖率定先后，否则页码会在相邻两块之间随机跳
TIE_MARGIN = 0.01

# 问句里的疑问/问法短语。中文问句的实体大多在这些词前面：“醉驾怎么处
# 罚”截断后只剩“醉驾”，才是该跟条文对字的部分；把“处罚”留在里面会让
# 每一款都沾上“罚”字，反而分不开
ASK_PART = re.compile(
    r"(怎么样|怎么|怎样|如何|什么|是多少|多少|多久|多长|哪些|几次|为什么|是否|能否|应该|可以|请问)"
)
# 虚词和疑问字不参与对字；“车”在法规里几乎每条都有，留着会让“高速公路
# 最高车速”靠一个“车”字被“不得进入高速公路”那种句子凑满覆盖
STOP_CHARS = set("怎么如什么多久吗呢吧啊了的地得是在有就和与及或我你他她它这那一个要会能可请问车")


def _question_chars(question):
    """问题里的实义汉字：删掉“怎么/多少/应该”这些问法词和虚词，不截断

    和 _entity_chars 的区别是不在问法词处截断：“机动车多久检验一次”要
    保留“检验”，截断后只剩“机动车”，跟讲“审验”的那一条就分不出来了。
    """
    core = ASK_PART.sub("", question)
    return {ch for ch in core if "\u4e00" <= ch <= "\u9fff"} - STOP_CHARS


def literal_cover(question, text):
    """问题实义字在块里的占比，用来给分数持平的块定先后"""
    chars = _question_chars(question)
    if not chars:
        return 0.0
    return sum(1 for ch in chars if ch in text) / len(chars)


# ====================== 文档索引 ======================
class DocIndex:
    """一份 PDF 的内存索引：切块、页码、向量三者一一对应"""

    def __init__(self, pdf_path=PDF_PATH, progress=None):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"找不到 PDF：{self.pdf_path}")

        embed_model = get_embed_model()
        doc = pymupdf.open(str(self.pdf_path))
        self.num_pages = doc.page_count
        pages = drop_toc_pages(drop_page_furniture(read_pages(doc, progress)))
        doc.close()
        if not pages:
            raise ValueError("PDF 里没有可用文本，可能是扫描版（需要 OCR）")

        self.chunks, self.chunk_pages, self.chunk_is_head = [], [], []
        body_text = {}
        for page in pages:
            text = drop_toc_lines(strip_page_number(
                "\n".join(line[0] for line in page["lines"]), page["no"]))
            if text.strip():
                body_text[page["no"]] = text

        articles = split_articles(list(body_text.items()))
        if articles:
            # 法规类：一条一块
            for text, page_no in articles:
                self.chunks.append(text)
                self.chunk_pages.append(page_no)
                self.chunk_is_head.append(False)
        else:
            body_size = body_font_size(pages)
            for page in pages:
                # 标题单独成块，再跟上第一句正文：短问题要的往往就是标题那半句
                for head, start in page_headings(page["lines"], body_size):
                    lead = heading_lead(page["lines"], start, body_size)
                    text = f"{head}\n{lead}".strip()
                    if len(re.sub(r"\s", "", text)) < 8:
                        continue                     # 光杆编号没有检索价值
                    self.chunks.append(text)
                    self.chunk_pages.append(page["no"])
                    self.chunk_is_head.append(True)
                for chunk in split_text(body_text.get(page["no"], "")):
                    if is_noise(chunk):
                        continue
                    self.chunks.append(chunk)
                    self.chunk_pages.append(page["no"])      # 页码从 1 开始
                    self.chunk_is_head.append(False)

        if not self.chunks:
            raise ValueError("PDF 里没有可用文本，可能是扫描版（需要 OCR）")

        self.embeddings = embed_model.encode(self.chunks, normalize_embeddings=True)
        self.nn = NearestNeighbors(
            n_neighbors=min(TOP_K, len(self.chunks)), metric="cosine"
        ).fit(self.embeddings)

    def search(self, question):
        embed_model = get_embed_model()
        query_emb = embed_model.encode(
            [QUERY_INSTRUCTION + question], normalize_embeddings=True
        )
        distances, indices = self.nn.kneighbors(query_emb)
        hits = []
        for d, i in zip(distances[0], indices[0]):
            score = 1 - float(d)
            if self.chunk_is_head[i]:
                score *= HEAD_PENALTY
            hits.append({
                "page": self.chunk_pages[i],
                "text": self.chunks[i],
                "score": round(score, 4),
            })
        hits.sort(key=lambda hit: -hit["score"])
        # 同一档分数用问题实义字的覆盖率再排一次：覆盖率高的块更可能真在讲这件事，
        # 打平时按原分数（sorted 是稳定的），不会把明显高分的那块挤下去
        start = 0
        while start < len(hits):
            end = start + 1
            while (end < len(hits)
                   and hits[start]["score"] - hits[end]["score"] < TIE_MARGIN):
                end += 1
            if end - start > 1:
                band = sorted(hits[start:end],
                              key=lambda hit: -literal_cover(question, hit["text"]))
                hits[start:end] = band
            start = end
        return hits


# ====================== 高亮定位 ======================
def _merge_boxes(boxes):
    """同一行里相邻字符的框合并成一块，高亮画几块而不是一个字一块"""
    rects = []
    for x0, y0, x1, y1 in boxes:
        if (rects and abs(rects[-1].y0 - y0) < 1.5 and abs(rects[-1].y1 - y1) < 1.5
                and x0 - rects[-1].x1 < 3):
            rects[-1].x1 = max(rects[-1].x1, x1)
            rects[-1].y0 = min(rects[-1].y0, y0)
            rects[-1].y1 = max(rects[-1].y1, y1)
        else:
            rects.append(pymupdf.Rect(x0, y0, x1, y1))
    return rects


def locate_chunk(page, chunk_text):
    """找出高亮文本在页面上的位置，返回矩形列表（PDF 坐标系）

    用 rawdict 取每个字符的坐标（不用 words：中文一个字也算词，按词框会把
    整行都框上，句子从行中间开始时前半行也跟着变黄）。把所有字符拼成去空白
    的字符流，拿高亮文本的前缀去找，命中后往后对齐到句子末尾，再按行合并
    字符框。
    """
    data = page.get_text("rawdict")
    chars, boxes = [], []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    if char["c"].isspace():
                        continue
                    chars.append(char["c"])
                    boxes.append(char["bbox"])
    flat = "".join(chars)
    needle = "".join(ch for ch in chunk_text if not ch.isspace())
    if not flat or not needle:
        return []

    # 先用句子开头定位（完整句子通常一次就中），实在不行才试切块留下的半句偏移；
    # 不足 8 个字的片段先不试：那种命中基本是碰巧，黄框会落到页面顶上的无关行。
    # 末尾的 len(needle) - start 是给短句留的：“步骤 3. 单击提交。”这种只有九个字，
    # 固定档位最小也是 10，不加就一直定位不到。
    # 全部落空时再放宽到 4 字：“LiCO的功能”这种标题去空白后只有 7 个字，
    # 不放宽就框都不画，但 4 字以下宁可不要
    for floor in (8, 4):
        for start in (0, 15, 30, 45):
            for size in (80, 60, 45, 35, 28, 20, 15, 10, len(needle) - start):
                if size < floor or start + size > len(needle):
                    continue
                pos = flat.find(needle[start:start + size])
                if pos < 0:
                    continue
                # 命中后往后对齐，把整句（最多 120 字）都框进来：只框前缀会漏句尾
                span = min(len(needle) - start, 120)
                same = 0
                while (same < span and pos + same < len(flat)
                       and flat[pos + same] == needle[start + same]):
                    same += 1
                end = pos + max(size, same)
                return _merge_boxes(boxes[pos:end])
    return []


# ====================== 回答 ======================
def build_prompt(question, hits):
    # 不在 prompt 里写“（第 N 页）”这种格式要求：小模型会把它当成要输出的一部分照抄。
    # 同样不让模型写“第 X 条”：小模型会把资料里的句子重新分配到编出来的条号上
    # （实测把第二十三条的内容写成“第十七条”），条号交给 collect_articles 从原文取。
    context = "\n\n".join(f"[第 {h['page']} 页] {h['text']}" for h in hits)
    return f"""资料：
{context}

问题：{question}

请只依据上面的资料，用两三句话直接回答结论。资料中只要提到了相关内容就据此回答，确实没有时再回答“{NOT_FOUND}”。回答里不需要写“第 X 条”编号。"""


def clean_answer(text):
    """去掉模型复述出来的占位符，如“第 N 页”"""
    return PLACEHOLDER.sub("", text).strip()


def attach_page(text, page):
    """页码由检索结果给出，避免小模型编造"""
    if PAGE_MARK.search(text):
        return text
    return f"{text}\n\n依据：第 {page} 页"


def collect_articles(hits, snippet_len=100):
    """从检索块里取出条号和原文，供答案末尾列出

    条号不让生成模型写：小模型会把资料里的句子重新安到编造的条号上，
    用户拿它去对照法条就会对不上。直接从原文开头取“第 X 条”，
    与 PDF 里的内容一一对应。
    """
    items = []
    for hit in hits:
        body = hit["text"].lstrip()
        match = ARTICLE_NUMBER.match(body)
        if not match:
            return []                     # 不是条文结构（普通文档），全部不附
        snippet = body[match.end():].lstrip()[:snippet_len].replace("\n", " ").strip()
        if len(body) - match.end() > snippet_len:
            snippet += "…"
        items.append((match.group(), snippet, hit["page"]))
    return items


def is_degenerate(answer):
    """小模型偶尔只吐一个页码（如“第 66 页”），去掉页码和标点后字数太少就算退化"""
    stripped = re.sub(r"[（(]?\s*第\s*[Nn\d]+\s*页\s*[）)]?", "", answer)
    stripped = re.sub(r"[^\w\u4e00-\u9fff]", "", stripped)
    return len(stripped) < 8


def is_not_found_sentence(sent):
    """判断一句是不是“资料中没有找到……”这种兜底句（含“关于 X 的相关内容”的扩写）

    “没找到具体周期，但规定了定期检验制度”这类句子里逗号后还有实质说明，
    不算兜底句；动词前面已经是一长串叙述的也不算。
    """
    match = NOT_FOUND_HINT.search(sent)
    if not match:
        return False
    if len(re.sub(r"[^\w\u4e00-\u9fff]", "", sent[:match.start()])) > 15:
        return False
    parts = [p for p in re.split(r"[，,；;]", sent[match.end():]) if p.strip()]
    if len(parts) >= 2:
        last = parts[-1]
        if len(re.sub(r"[^\w\u4e00-\u9fff]", "", last)) > 8 and not NOT_FOUND_HINT.search(last):
            return False
    return True


def strip_not_found(text):
    """剔除“资料中没有找到……”兜底句，返回 (剩余文本, 是否整段只剩兜底)"""
    sentences = [s for s in re.split(r"(?<=[。！？\n])", text) if s.strip()]
    kept = [s for s in sentences if not is_not_found_sentence(s)]
    real = "".join(kept).strip()
    if sentences and not real:
        return "", True
    return real or text, False


def split_sentences(text, limit=100):
    """把检索块切成能标黄的短段

    只按“。！？”切不够：代码块、列表行和英文段落常常整段没有一个
    中文句号，400 字的块会切不碎，标黄时一整片都变黄（用户要的是
    “有关的第一段”）。没句号就退到逗号，再不行就硬切。
    """
    parts = re.split(r"(?<=[。！？；\n])", text)
    # 英文句点跟着空格才算句末，避开 3.5 / e.g. 这类；代码块常靠这个切开
    parts = [sub for p in parts if p
             for sub in (re.split(r"(?<=[.!?;])\s", p) if len(p) > limit else [p])]
    parts = [sub for p in parts if p
             for sub in (re.split(r"(?<=[，,、：:])", p) if len(p) > limit else [p])]
    out = []
    for part in parts:
        part = part.strip()
        while len(part) > limit:              # 连逗号都没有的长代码行
            out.append(part[:limit])
            part = part[limit:]
        if part:
            out.append(part)
    return out


def _entity_chars(question):
    """问题里该和条文对字的那部分：截掉“怎么 / 多少 / 应该”这些问法词"""
    core = ASK_PART.split(question)[0].strip("，,、。？?！! ")
    if len(core) < 2:                        # 问法词在开头，改成删掉它
        core = ASK_PART.sub("", question)
    return {ch for ch in core if "\u4e00" <= ch <= "\u9fff"} - STOP_CHARS


def char_match(question, sentence):
    """问题实体在句中出现的比例和紧凑度，返回 (覆盖率, 命中跨度)

    bge-small 对同一条里“饮酒 / 再次饮酒 / 醉酒”几款只差 0.001，块内比
    不出用户问的是哪一款；字面却能比：问“醉驾”只有醉酒那款含“醉”字，
    问“饮酒后驾驶机动车”是饮酒那几款全含。跨度是命中字首次出现位置的
    差，越小说明它们越集中，不是东一个字西一个字碰巧凑上的。
    """
    chars = _entity_chars(question)
    if not chars:
        return 0.0, 0
    hits = [sentence.find(ch) for ch in chars if ch in sentence]
    if not hits:
        return 0.0, 0
    return len(hits) / len(chars), max(hits) - min(hits)


def pick_highlights(question, hits):
    """给每个检索块挑出最相关的一段，供 PDF 标黄

    之前法规整条标黄，一条常有饮酒 / 营运 / 醉酒好几款，页面上一大片黄
    反而看不出答案落在哪句。现在每个块只标一段：切句后法规按问题实体
    的字面覆盖挑出对应的那一款（持平取字更集中的，再相同取靠前的），
    普通文档取向量分最高的一段。
    """
    groups, legal_flags = [], []
    for hit in hits:
        text = hit["text"].lstrip()
        is_legal = bool(ARTICLE_NUMBER.match(text))
        if is_legal:
            # 法规是分段排的，行尾不带标点，按行切会把“法律”切成“法/律”，
            # 选出来的段就成了半句；先接回整段再切句
            text = re.sub(r"\s*\n\s*", "", text)
        groups.append([s for s in split_sentences(text) if len(s) >= 8])
        legal_flags.append(is_legal)
    flat = [seg for group in groups for seg in group]
    if not flat:
        for hit in hits:
            hit["highlight"] = hit["text"][:120]
        return hits

    embed_model = get_embed_model()
    query_emb = embed_model.encode([QUERY_INSTRUCTION + question], normalize_embeddings=True)
    scores = embed_model.encode(flat, normalize_embeddings=True) @ query_emb[0]

    cursor = 0
    for hit, group, is_legal in zip(hits, groups, legal_flags):
        if not group:
            hit["highlight"] = hit["text"][:120]
            continue
        scores_group = scores[cursor:cursor + len(group)]
        cursor += len(group)
        if not is_legal:
            hit["highlight"] = group[int(scores_group.argmax())]
            continue
        # 同一条里常并列好几款（饮酒 / 再次饮酒 / 醉酒 / 营运…），向量分
        # 几乎一样，靠字面挑：问题实体对得上字最多的一款；一样多时取字
        # 更集中的（东一个字西一个字碰巧凑上的不算）；再相同取靠前的——
        # 用户问“怎么处罚”要的是基本情形那一款，不是后面加重的特殊情形
        matched = [char_match(question, s) for s in group]
        if max(m[0] for m in matched) > 0:
            best = max(range(len(group)), key=lambda i: (
                round(matched[i][0], 2), -matched[i][1], -i))
            hit["highlight"] = group[best]
        else:                                # 一个字都对不上，退回向量分
            hit["highlight"] = group[int(scores_group.argmax())]
    return hits


def extractive_answer(question, hits, top_n=3):
    """兜底：不靠生成模型，用向量给句子打分，直接摘手册原句"""
    embed_model = get_embed_model()
    sentences, seen = [], set()
    for hit in hits:
        for sent in re.split(r"(?<=[。！？；])", hit["text"]):
            sent = sent.strip()
            if len(sent) < 10 or sent in seen:
                continue
            if NAV_SENTENCE.search(sent):
                continue                     # 丢掉“请参阅第 59 页”这类无信息量的导航句
            seen.add(sent)
            sentences.append((sent, hit["page"]))
    if not sentences:
        return hits[0]["text"][:300]

    query_emb = embed_model.encode([QUERY_INSTRUCTION + question], normalize_embeddings=True)
    sent_emb = embed_model.encode([s for s, _ in sentences], normalize_embeddings=True)
    scores = sent_emb @ query_emb[0]
    picked = sorted(scores.argsort()[::-1][:top_n])
    return "\n".join(
        f"{i + 1}. {sentences[j][0]}（第 {sentences[j][1]} 页）" for i, j in enumerate(picked)
    )


def answer_question(question, index, server):
    """检索 + 生成，必要时退化为抽取式。返回 (答案, hits, 模式, 耗时)"""
    started = time.time()
    hits = pick_highlights(question, index.search(question))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, hits)},
    ]
    text, usage = server.chat(messages, max_tokens=MAX_NEW_TOKENS, temperature=0.0)
    text = clean_answer(text)
    # 模型常把兜底句扩写成“资料中没有找到关于 X 的相关内容”，也会先答结论再
    # 照抄一遍：统一按句剔除；整段只剩兜底句时直接给标准句、不返回检索块，
    # 否则界面会把一堆不相关段落列出来，看着像答上了。
    text, not_found = strip_not_found(text)
    if not_found:
        return f"{NOT_FOUND}。", [], "none", round(time.time() - started, 1), usage

    mode = "llm"
    if is_degenerate(text):
        text = extractive_answer(question, hits)
        mode = "extractive"
    else:
        items = collect_articles(hits)
        if not items:
            text = attach_page(text, hits[0]["page"])
        else:
            related = "\n".join(f"- {num}：{snip}（第 {page} 页）"
                               for num, snip, page in items)
            text = f"{text}\n\n相关规定：\n{related}"

    return text, hits, mode, round(time.time() - started, 1), usage