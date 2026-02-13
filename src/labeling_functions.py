from __future__ import annotations

import re
from typing import Optional, List, Tuple

# Label scheme: 1 = CLAIM, 0 = NOT_CLAIM, -1 = ABSTAIN
ABSTAIN = -1
CLAIM = 1
NOT_CLAIM = 0


_CLAIM_PATTERNS = [
    r"\b(we|this study|our results|our findings)\b.*\b(show|demonstrate|indicate|suggest|reveal)\b",
    r"\b(results|findings)\b.*\b(show|demonstrate|indicate|suggest)\b",
    r"\b(improve|outperform|achieve|increase|reduce)\b.*\b(accuracy|auc|f1|precision|recall|performance)\b",
    r"\b(significant|significantly)\b",
    r"\b(we propose|we present|we introduce)\b",
    r"\b(can|could|may)\b.*\b(help|enable|support|facilitate)\b",
]

_NONCLAIM_PATTERNS = [
    r"\b(in this paper|in this work)\b.*\b(we describe|we review|we survey)\b",
    r"\b(background|motivation|objective)\b",
    r"\b(dataset|data set|cohort)\b.*\b(consists of|comprises|contains)\b",
    r"\b(methods?|materials and methods?)\b",
    r"\b(we used|we performed|we conducted)\b",
]

_RESULT_VERB_PATTERNS = [
    r"\bdemonstrate(s|d|ing)?\b",
    r"\bindicate(s|d|ing)?\b",
    r"\bshow(s|ed|ing)?\b",
    r"\bachiev(e|ed|es|ing)\b",
    r"\bresult(s|ed|ing)? in\b",
    r"\byield(s|ed|ing)?\b",
    r"\bimprov(e|ed|es|ing|ement)\b",
    r"\boutperform(s|ed|ing)?\b",
    r"\breduc(e|ed|es|ing|tion)\b",
]

_METRIC_PATTERNS = [
    r"\baccuracy\b",
    r"\bauc\b",
    r"\bauroc\b",
    r"\bf1(?:-score)?\b",
    r"\bprecision\b",
    r"\brecall\b",
    r"\bsensitivity\b",
    r"\bspecificity\b",
    r"\bdice\b",
    r"\bjaccard\b",
    r"\biou\b",
    r"\bhazard ratio\b",
    r"\bodds ratio\b",
    r"\bhr\s*[:=]?\s*\d+(\.\d+)?\b",
    r"\bor\s*[:=]?\s*\d+(\.\d+)?\b",
    r"\baor\s*[:=]?\s*\d+(\.\d+)?\b",
    r"\bsmd\b",
    r"\br\s*=\s*[-+]?\d+(\.\d+)?\b",
    r"\bp\s*[<=>]\s*0?\.\d+\b",
    r"\bci\b",
    r"\bic50\b",
    r"\bmedian survival\b",
    r"\bp[- ]?value\b",
]

_COMPARISON_PATTERNS = [
    r"\bcompared to\b",
    r"\bversus\b",
    r"\bbaseline\b",
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\boutperform(s|ed|ing)?\b",
]

_NOVELTY_PATTERNS = [
    r"\bnovel\b",
    r"\bnew\b",
    r"\bfirst\b",
    r"\bfor the first time\b",
    r"\bwe propose\b",
    r"\bwe present\b",
    r"\bwe introduce\b",
    r"\bwe developed\b",
    r"\bwe designed\b",
    r"\bour work demonstrates\b",
    r"\bour findings\b",
]

_BACKGROUND_SUPPRESSION_PATTERNS = [
    r"\b(is|are)\b.*\b(aggressive|common|rare|deadly|incurable)\b",
    r"\b(background|introduction|objective)\b",
    r"\b(remains a challenge|unmet need|limited treatment options)\b",
    r"\b(we reviewed|we describe|this review)\b",
    r"\b(is|are)\s+a\s+common\b",
    r"\b(is|are)\s+the\s+most\s+common\b",
    r"\bis a significant challenge\b",
    r"\bremains incurable\b",
    r"\bdespite advances\b",
    r"\bhas poor prognosis\b",
]

_METHODS_PATTERNS = [
    r"\b(we used|we performed|we conducted|retrospective|prospective)\b",
    r"\b(training set|validation set|test set|cross[- ]validation|five[- ]fold|k[- ]fold)\b",
    r"\b(assay|pipeline|preprocessing|segmentation|feature extraction)\b",
]

_CASE_OBSERVATION_PATTERNS = [
    r"\b\d{1,3}-year-old\b",
    r"\bpresented to (the )?(hospital|emergency department)\b",
    r"\bcase report\b",
    r"\bwork-up revealed\b",
]

_SECTION_PATTERNS = {
    "background": [r"^(background|introduction|objective|objectives|rationale)\b[:\s-]*"],
    "methods": [r"^(methods|materials and methods|design|patients and methods)\b[:\s-]*"],
    "results": [r"^(results|findings|observations)\b[:\s-]*"],
    "conclusion": [r"^(conclusion|conclusions|in conclusion|overall)\b[:\s-]*"],
}

_STRUCTURED_SECTION_HEADER_PATTERNS = {
    "background": re.compile(
        r"^\s*(background|introduction|objective|objectives|rationale|purpose)\s*[:\.\-\s]*",
        re.I,
    ),
    "methods": re.compile(
        r"^\s*(methods|materials and methods|design|patients and methods)\s*[:\.\-\s]*",
        re.I,
    ),
    "results": re.compile(r"^\s*(results|findings|observations)\s*[:\.\-\s]*", re.I),
    "conclusion": re.compile(r"^\s*(conclusion|conclusions|in conclusion|overall)\s*[:\.\-\s]*", re.I),
}

_VAGUE_PHRASES = [
    r"\bmay contribute\b",
    r"\bprovides? insight\b",
    r"\bhighlights? the importance\b",
    r"\bunderscores?\b",
    r"\brepresents? a novel contribution\b",
    r"\bholds? promise\b",
    r"\bsuggests? that\b",
    r"\bcould be useful\b",
    r"\bmight enable\b",
    r"\bfuture studies are needed\b",
    r"\bgrowing need\b",
]

_BACKGROUND_DEFINITION_PATTERNS = [
    r"^\s*(a|an|the)?\s*brain\s+(is|are)\b",
    r"^\s*(a|an|the)?\s*tumou?rs?\s+(is|are)\b",
    r"^\s*(a|an|the)?\s*glioma(s)?\s+(is|are)\b",
    r"^\s*(a|an|the)?\s*glioblastoma(s)?\s+(is|are)\b",
    r"^\s*(a|an|the)?\s*brain tumou?rs?\s+(is|are)\b",
    r"^\s*e\s+human\s+brain\b",  # noisy OCR variant seen in this corpus
    r"\bcerebral cortex\b.*\bcerebellum\b",
    r"\bis known as\b.*\b(tumou?r|neoplasm)\b",
    r"\bbrain tumou?r\b.*\b(abnormal cell accumulation|neoplasm|mass|lump)\b",
    r"^\s*(benign|malignant)\s+tumou?rs?\b",
    r"^\s*brain tumou?rs?\s+are\b",
    r"^\s*brain tumou?r diagnosis involves\b",
    r"^\s*tumou?rs?\s+can\s+be\s+found\b",
    r"^\s*understanding\s+(the\s+)?(growth(\s+and\s+behavio[u]?r)?|behavio[u]?r)\s+of\s+(these\s+)?tumou?rs?\b",
    r"\bthere\s+is\s+a\s+growing\s+need\b",
]

_GENERIC_CONCLUSION_PATTERNS = [
    r"\bmay contribute\b",
    r"\bholds? promise\b",
    r"\bsuggests? that\b",
    r"\brepresents? a novel contribution\b",
    r"\bgrowing\W+need\b",
]

_STAT_INDICATOR_PATTERNS = [
    r"\b\d+(\.\d+)?\s*%\b",
    r"\bp\s*[<=>]\s*0?\.\d+\b",
    r"\bci\b",
    r"\bhazard ratio\b",
    r"\bodds ratio\b",
    r"\bhr\s*[:=]?\s*\d+(\.\d+)?\b",
    r"\bor\s*[:=]?\s*\d+(\.\d+)?\b",
    r"\baor\s*[:=]?\s*\d+(\.\d+)?\b",
    r"\bsmd\b",
    r"\br\s*=\s*[-+]?\d+(\.\d+)?\b",
    r"\bauc\b",
    r"\bic50\b",
    r"\bsensitivity\b",
    r"\bspecificity\b",
    r"\baccuracy\b",
    r"\bdice\b",
    r"\bjaccard\b",
]

_CLINICAL_ENDPOINT_PATTERNS = [
    r"\bsurvival\b",
    r"\brecurrence\b",
    r"\btumou?r volume\b",
    r"\bclearance\b",
    r"\boverall survival\b",
    r"\bmortality\b",
    r"\bprogression\b",
    r"\binhibition\b",
    r"\belimination\b",
]

_MECHANISM_PATTERNS = [
    r"\bpathway\b",
    r"\bsignall?ing\b",
    r"\bmechanis(m|tic)\b",
    r"\bmediated by\b",
    r"\bupregulat(ed|es|ion|ing)\b",
    r"\bdownregulat(ed|es|ion|ing)\b",
    r"\bphosphorylation\b",
]

_REVIEW_PATTERNS = [
    r"\breview\b",
    r"\bmeta[- ]analysis\b",
    r"\bsystematic review\b",
    r"\bnarrative review\b",
    r"\bscientometric\b",
    r"\bbibliometric\b",
]

_CASE_REPORT_PATTERNS = [
    r"\bcase report\b",
    r"\billustrative case\b",
]

_REVIEW_PRIORITY_PATTERNS = [
    r"\bpooled\b",
    r"\bprevalence\b",
    r"\bsynthesis\b",
    r"\bmeta[- ]analysis\b",
]

_CASE_PRIORITY_PATTERNS = [
    r"\brare\b",
    r"\bdiagnostic pitfall\b",
    r"\bmisdiagnos(is|ed)\b",
]

_SECTION_MULTIPLIER = {
    "background": 0.70,
    "methods": 0.80,
    "results": 1.20,
    "conclusion": 1.25,
    "other": 1.00,
}

_INCOMPLETE_ENDINGS = (
    " of",
    " to",
    " and",
    " with",
    " for",
    " in",
    " the",
    " a",
    " an",
)

def lf_claim_pattern(sentence: str) -> int:
    s = sentence.lower()
    for p in _CLAIM_PATTERNS:
        if re.search(p, s):
            return CLAIM
    return ABSTAIN

def lf_nonclaim_pattern(sentence: str) -> int:
    s = sentence.lower()
    for p in _NONCLAIM_PATTERNS:
        if re.search(p, s):
            return NOT_CLAIM
    return ABSTAIN

def weak_claim_score(sentence: str) -> float:
    """
    Simple score in [0, 1] based on heuristics.
    Used as a fallback if SciBERT model isn't available.
    """
    votes = []
    v1 = lf_claim_pattern(sentence)
    v2 = lf_nonclaim_pattern(sentence)
    if v1 != ABSTAIN:
        votes.append(v1)
    if v2 != ABSTAIN:
        votes.append(v2)

    if not votes:
        return 0.35  # neutral prior

    # if any NOT_CLAIM vote, reduce score
    if NOT_CLAIM in votes and CLAIM not in votes:
        return 0.10
    if CLAIM in votes and NOT_CLAIM not in votes:
        return 0.80
    # conflict
    return 0.45


def has_strong_result_verb(sentence: str) -> bool:
    s = (sentence or "").lower()
    return any(re.search(p, s) for p in _RESULT_VERB_PATTERNS)


def has_ml_metric_signal(sentence: str) -> bool:
    s = (sentence or "").lower()
    if any(re.search(p, s) for p in _METRIC_PATTERNS):
        return True
    return bool(re.search(r"\b\d+(\.\d+)?\s*%\b", s))


def has_statistical_indicator(sentence: str) -> bool:
    s = (sentence or "").lower()
    return any(re.search(p, s) for p in _STAT_INDICATOR_PATTERNS)


def has_clinical_endpoint(sentence: str) -> bool:
    s = (sentence or "").lower()
    return any(re.search(p, s) for p in _CLINICAL_ENDPOINT_PATTERNS)


def has_mechanism_signal(sentence: str) -> bool:
    s = (sentence or "").lower()
    return any(re.search(p, s) for p in _MECHANISM_PATTERNS)


def has_vague_scientific_language(sentence: str) -> bool:
    s = (sentence or "").lower()
    return any(re.search(p, s) for p in _VAGUE_PHRASES)


def has_strong_claim_starter(sentence: str) -> bool:
    s = (sentence or "").lower().strip()
    return bool(re.match(r"^(we found|we demonstrate|we showed|our results show)\b", s))


def has_comparison_signal(sentence: str) -> bool:
    s = (sentence or "").lower()
    return any(re.search(p, s) for p in _COMPARISON_PATTERNS)


def has_novelty_cue(sentence: str) -> bool:
    s = (sentence or "").lower()
    return any(re.search(p, s) for p in _NOVELTY_PATTERNS)


def is_background_sentence(sentence: str) -> bool:
    s = (sentence or "").lower()
    return any(re.search(p, s) for p in _BACKGROUND_SUPPRESSION_PATTERNS)


def is_background_definition_sentence(sentence: str) -> bool:
    s = (sentence or "").lower().strip()
    return any(re.search(p, s) for p in _BACKGROUND_DEFINITION_PATTERNS)


def is_generic_conclusion_sentence(sentence: str) -> bool:
    s = (sentence or "").lower().strip()
    return any(re.search(p, s) for p in _GENERIC_CONCLUSION_PATTERNS)


def is_methods_sentence(sentence: str) -> bool:
    s = (sentence or "").lower()
    return any(re.search(p, s) for p in _METHODS_PATTERNS)


def is_case_observation_sentence(sentence: str) -> bool:
    s = (sentence or "").lower()
    return any(re.search(p, s) for p in _CASE_OBSERVATION_PATTERNS)


def detect_document_mode(title: Optional[str], abstract: Optional[str] = None) -> str:
    s = f"{title or ''} {abstract or ''}".lower()
    if any(re.search(p, s) for p in _REVIEW_PATTERNS):
        return "review"
    if any(re.search(p, s) for p in _CASE_REPORT_PATTERNS):
        return "case_report"
    return "default"


def _section_header_label(sentence: str) -> Optional[str]:
    s = (sentence or "").strip()
    for section, rx in _STRUCTURED_SECTION_HEADER_PATTERNS.items():
        if rx.search(s):
            return section
    return None


def infer_abstract_sections(sentences: List[str]) -> Tuple[List[str], bool]:
    if not sentences:
        return [], False
    sections: List[str] = []
    current = "other"
    seen = set()
    explicit = 0
    for i, s in enumerate(sentences):
        label = _section_header_label(s)
        if label:
            current = label
            seen.add(label)
            explicit += 1
            sections.append(label)
            continue
        if current != "other":
            sections.append(current)
        else:
            sections.append(detect_sentence_section(s, idx=i, total=len(sentences)))
    has_structured = explicit >= 2 and ("results" in seen or "conclusion" in seen)
    return sections, has_structured


def endpoint_priority_tier(sentence: str) -> int:
    """
    Higher value means more likely to be a primary outcome-level claim.
    5: survival/recurrence/tumor volume
    4: clinical accuracy/diagnostic metrics
    3: drug effect (inhibition/elimination)
    2: mechanism
    1: general discussion
    """
    s = (sentence or "").lower()
    if re.search(r"\b(survival|recurrence|tumou?r volume|overall survival|mortality|progression)\b", s):
        return 5
    if re.search(
        r"\b(accuracy|auc|auroc|sensitivity|specificity|dice|jaccard|iou|f1|precision|recall|hazard ratio|odds ratio|smd|ic50)\b",
        s,
    ):
        return 4
    if re.search(r"\b(inhibition|elimination|suppression|reduced growth|tumor shrinkage)\b", s):
        return 3
    if has_mechanism_signal(sentence):
        return 2
    return 1


def detect_sentence_section(sentence: str, idx: int = 0, total: int = 1) -> str:
    s = (sentence or "").strip().lower()
    for section, patterns in _SECTION_PATTERNS.items():
        if any(re.search(p, s) for p in patterns):
            return section
    if total > 1:
        rel = idx / max(1, total - 1)
        if rel < 0.25:
            return "background"
        if rel > 0.75:
            return "conclusion"
    return "other"


def is_incomplete_sentence(sentence: str) -> bool:
    s = (sentence or "").strip()
    if len(s) < 25:
        return True
    if s.lower().endswith(_INCOMPLETE_ENDINGS):
        return True
    if not s.endswith((".", "!", "?")) and len(s) > 80:
        return True
    return False


def sentence_quality_score(sentence: str) -> float:
    s = (sentence or "").strip()
    tokens = s.split()
    n = len(tokens)
    if n < 8:
        return 0.20
    if n > 60:
        return 0.70
    if 12 <= n <= 40:
        return 1.00
    return 0.85


def should_filter_candidate(sentence: str, section: str = "other") -> bool:
    if is_background_definition_sentence(sentence):
        return True
    if is_incomplete_sentence(sentence):
        return True
    if section == "background" and not (
        has_strong_result_verb(sentence) or has_ml_metric_signal(sentence) or has_novelty_cue(sentence)
    ):
        return True
    if is_methods_sentence(sentence) and not (has_ml_metric_signal(sentence) or has_strong_result_verb(sentence)):
        return True
    if is_case_observation_sentence(sentence) and not has_strong_result_verb(sentence):
        return True
    if has_vague_scientific_language(sentence) and not (
        has_statistical_indicator(sentence) or has_clinical_endpoint(sentence)
    ):
        return True
    return False


def engineered_claim_score(
    sentence: str,
    alpha: float = 0.50,
    beta: float = 0.20,
    gamma: float = 0.20,
    delta: float = 0.10,
) -> float:
    p_claim = weak_claim_score(sentence)
    r = 1.0 if has_strong_result_verb(sentence) else 0.0
    m = 1.0 if has_ml_metric_signal(sentence) else 0.0
    c = 1.0 if has_comparison_signal(sentence) else 0.0
    score = alpha * p_claim + beta * r + gamma * m + delta * c
    return max(0.0, min(1.0, score))


def rerank_claim_score(
    sentence: str,
    p_claim: Optional[float] = None,
    section: str = "other",
    doc_mode: str = "default",
    alpha: float = 0.50,
    beta: float = 0.20,
    gamma: float = 0.20,
    delta: float = 0.10,
) -> float:
    base = weak_claim_score(sentence) if p_claim is None else float(p_claim)
    r = 1.0 if has_strong_result_verb(sentence) else 0.0
    m = 1.0 if has_ml_metric_signal(sentence) else 0.0
    c = 1.0 if has_comparison_signal(sentence) else 0.0
    n = 1.0 if has_novelty_cue(sentence) else 0.0
    novelty_boost = 0.08 * n
    starter_boost = 0.10 if has_strong_claim_starter(sentence) else 0.0
    stat_boost = 0.22 if has_statistical_indicator(sentence) else 0.0
    endpoint_boost = 0.22 if has_clinical_endpoint(sentence) else 0.0
    mechanism_penalty = 0.18 if (has_mechanism_signal(sentence) and not has_clinical_endpoint(sentence)) else 0.0
    vague_penalty = 0.30 if has_vague_scientific_language(sentence) else 0.0
    background_penalty = 0.24 if (section == "background" or is_background_sentence(sentence)) else 0.0
    methods_penalty = 0.12 if is_methods_sentence(sentence) else 0.0
    case_penalty = 0.14 if is_case_observation_sentence(sentence) else 0.0
    incomplete_penalty = 0.28 if is_incomplete_sentence(sentence) else 0.0
    doc_mode_adjustment = 0.0
    s = (sentence or "").lower()
    if doc_mode == "review":
        if any(re.search(p, s) for p in _REVIEW_PRIORITY_PATTERNS):
            doc_mode_adjustment += 0.18
        if has_mechanism_signal(sentence):
            doc_mode_adjustment -= 0.10
    elif doc_mode == "case_report":
        if any(re.search(p, s) for p in _CASE_PRIORITY_PATTERNS):
            doc_mode_adjustment += 0.20
        if re.search(r"\b(patient (improved|was discharged|recovered)|improved)\b", s):
            doc_mode_adjustment -= 0.15
    quality = sentence_quality_score(sentence)
    score = (
        alpha * base
        + beta * r
        + gamma * m
        + delta * c
        + novelty_boost
        + starter_boost
        + stat_boost
        + endpoint_boost
        + doc_mode_adjustment
        - background_penalty
        - methods_penalty
        - case_penalty
        - incomplete_penalty
        - vague_penalty
        - mechanism_penalty
    )
    if section in {"results", "conclusion"}:
        score += 0.12
    if section in {"background", "methods"}:
        score -= 0.20
    score *= _SECTION_MULTIPLIER.get(section, 1.0)
    score *= quality
    return max(0.0, min(1.0, score))
