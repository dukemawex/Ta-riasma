#!/usr/bin/env python3
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cosine
from tabulate import tabulate


RESULTS_DIR = Path(__file__).resolve().parent / "results"
PARAPHRASE_CACHE_PATH = RESULTS_DIR / "paraphrase_cache.json"
EMBED_CACHE_PATH = RESULTS_DIR / "embedding_cache_dup.json"
RESULT_JSON_PATH = RESULTS_DIR / "duplicate_results.json"
REPORT_MD_PATH = RESULTS_DIR / "duplicate_report.md"

EMBEDDING_MODEL = "text-embedding-004"
PARAPHRASE_MODEL = "claude-sonnet-4-20250514"
THRESHOLDS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
MAX_RETRIES = 5
INITIAL_BACKOFF = 2
PARAPHRASE_RATE_SECONDS = 3
AGENTROUTER_OPENAI_BASE_URL = "https://agentrouter.org/v1"


class EvaluationClient:
    def __init__(self):
        self._anthropic_client = None
        self._gemini_client = None
        self._openai_client = None
        self._embedding_mode = "unknown"
        self._last_paraphrase_ts = 0.0
        gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if gemini_key:
            try:
                import google.generativeai as genai  # type: ignore

                genai.configure(api_key=gemini_key)
                self._gemini_client = genai
                self._embedding_mode = "gemini"
            except Exception as exc:
                print(f"Falling back from google-generativeai SDK: {exc}")

        if self._embedding_mode == "unknown":
            anthropic_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
            if not anthropic_key:
                raise RuntimeError(
                    "Set GEMINI_API_KEY for direct Gemini embeddings, or set ANTHROPIC_API_KEY to route embeddings via AgentRouter OpenAI endpoint."
                )
            try:
                from openai import OpenAI  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "openai SDK is required for AgentRouter-routed embeddings when GEMINI_API_KEY is not set."
                ) from exc
            self._openai_client = OpenAI(api_key=anthropic_key, base_url=AGENTROUTER_OPENAI_BASE_URL)
            self._embedding_mode = "agentrouter-openai"

    def _with_backoff(self, fn):
        backoff = INITIAL_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return fn()
            except Exception as exc:
                text = str(exc).lower()
                is_rate = "429" in text or "rate" in text or "too many" in text
                if not is_rate or attempt == MAX_RETRIES:
                    raise
                print(f"Rate limit detected; retrying in {backoff}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError("Backoff exhausted unexpectedly")

    def get_embedding(self, text: str, model: str) -> List[float]:
        def call():
            if self._embedding_mode == "gemini" and self._gemini_client is not None:
                resp = self._gemini_client.embed_content(model=model, content=text)
                emb = resp.get("embedding")
                if emb:
                    return emb
                raise RuntimeError(f"Unexpected Gemini embedding response shape: {resp}")

            if self._openai_client is not None:
                resp = self._openai_client.embeddings.create(model=model, input=text)
                if resp.data:
                    return resp.data[0].embedding
                raise RuntimeError("OpenAI-compatible embedding response contained no vectors")

            raise RuntimeError("Embedding client is not configured")

        return self._with_backoff(call)

    def generate_paraphrases(self, base_text: str) -> Dict[str, str]:
        elapsed = time.time() - self._last_paraphrase_ts
        if elapsed < PARAPHRASE_RATE_SECONDS:
            time.sleep(PARAPHRASE_RATE_SECONDS - elapsed)

        prompt = (
            "Rewrite the proposal below into three versions with the same meaning and key facts. "
            "Return ONLY valid JSON in this exact shape: "
            "{\"formal\": \"...\", \"concise\": \"...\", \"narrative\": \"...\"}. "
            "Rules: formal=academic register; concise=about 60% of original length; "
            "narrative=storytelling framing.\n\n"
            f"Proposal:\\n{base_text}"
        )

        def call():
            if self._anthropic_client is None:
                try:
                    import anthropic  # type: ignore
                except Exception as exc:
                    raise RuntimeError(
                        "anthropic SDK is required for Claude paraphrase generation. Install it with pip."
                    ) from exc
                self._anthropic_client = anthropic.Anthropic()
            resp = self._anthropic_client.messages.create(
                model=PARAPHRASE_MODEL,
                system="You output strict JSON only.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1200,
            )
            text_content = ""
            for block in resp.content:
                if block.type == "text":
                    text_content += block.text
            if not text_content:
                raise RuntimeError(f"Unexpected Anthropic response shape: {resp}")
            return parse_json_block(text_content)

        result = self._with_backoff(call)
        self._last_paraphrase_ts = time.time()
        return result


def parse_json_block(text: str) -> Dict[str, str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        obj = json.loads(text)
        return {
            "formal": obj["formal"],
            "concise": obj["concise"],
            "narrative": obj["narrative"],
        }
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        obj = json.loads(match.group(0))
        return {
            "formal": obj["formal"],
            "concise": obj["concise"],
            "narrative": obj["narrative"],
        }


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def sentence_split(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [c.strip() for c in chunks if c.strip()]


def sentence_join(sentences: List[str]) -> str:
    return " ".join(s.strip() for s in sentences if s.strip())


SYNONYMS = {
    "provide": "supply",
    "community": "neighbourhood",
    "support": "assist",
    "funding": "financing",
    "project": "initiative",
    "implement": "execute",
    "develop": "create",
    "ensure": "guarantee",
    "improve": "enhance",
    "access": "availability",
}


def synonym_variant(text: str) -> str:
    replaced = text
    candidates = []
    for src in SYNONYMS:
        if re.search(rf"\b{re.escape(src)}\b", replaced, flags=re.IGNORECASE):
            candidates.append(src)
    if not candidates:
        return text

    target_count = min(len(candidates), random.randint(5, 8))
    chosen = random.sample(candidates, k=target_count)
    for src in chosen:
        dst = SYNONYMS[src]
        replaced = re.sub(
            rf"\b{re.escape(src)}\b",
            dst,
            replaced,
            count=1,
            flags=re.IGNORECASE,
        )
    return replaced


def to_participle(verb: str) -> str:
    base = verb.lower()
    irregular = {
        "be": "been",
        "begin": "begun",
        "break": "broken",
        "build": "built",
        "buy": "bought",
        "come": "come",
        "do": "done",
        "draw": "drawn",
        "drink": "drunk",
        "drive": "driven",
        "eat": "eaten",
        "fall": "fallen",
        "find": "found",
        "get": "gotten",
        "give": "given",
        "go": "gone",
        "grow": "grown",
        "have": "had",
        "know": "known",
        "make": "made",
        "read": "read",
        "run": "run",
        "see": "seen",
        "send": "sent",
        "take": "taken",
        "write": "written",
    }
    if base in irregular:
        return irregular[base]
    if base.endswith("e"):
        return f"{base}d"
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        return f"{base[:-1]}ied"
    if (
        len(base) >= 3
        and base[-1] not in "aeiouwxy"
        and base[-2] in "aeiou"
        and base[-3] not in "aeiou"
    ):
        return f"{base}{base[-1]}ed"
    return f"{base}ed"


def passive_transform_sentence(sentence: str) -> str:
    original = sentence.strip()
    terminal = "." if original.endswith(".") else ""
    core = original[:-1] if terminal else original

    modal_match = re.match(
        r"^\s*([A-Z][A-Za-z0-9 ,'-]+?)\s+(will|can|may|should|must)\s+([A-Za-z]+)\s+(.+)$",
        core,
    )
    if modal_match:
        subj, modal, verb, obj = modal_match.groups()
        obj = obj.strip()
        return f"{obj.capitalize()} {modal} be {to_participle(verb)} by {subj}{terminal}"

    simple_match = re.match(r"^\s*([A-Z][A-Za-z0-9 ,'-]+?)\s+([a-z]+s?)\s+(.+)$", core)
    if simple_match:
        subj, verb, obj = simple_match.groups()
        obj = obj.strip()
        return f"{obj.capitalize()} is {to_participle(verb)} by {subj}{terminal}"

    return sentence


def passive_variant(text: str) -> str:
    parts = sentence_split(text)
    if not parts:
        return text
    transformed = []
    for idx, sent in enumerate(parts):
        if idx < 2:
            transformed.append(passive_transform_sentence(sent))
        else:
            transformed.append(sent)
    return sentence_join(transformed)


def shuffle_variant(text: str) -> str:
    parts = sentence_split(text)
    if len(parts) <= 1:
        return text
    random.shuffle(parts)
    return sentence_join(parts)


def cosine_sim(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    if np.linalg.norm(va) == 0 or np.linalg.norm(vb) == 0:
        return 0.0
    return float(1 - cosine(va, vb))


def build_base_proposals() -> List[dict]:
    return [
        # climate
        {"id": "P01", "domain": "climate", "text": "This climate resilience proposal will map heat-risk blocks and create neighborhood cooling corridors. The municipality will plant drought-tolerant trees along school routes and market streets. We will provide open monitoring dashboards so residents can track canopy growth and maintenance quality. Local youth teams will support watering schedules during extreme heat months."},
        {"id": "P02", "domain": "climate", "text": "Our adaptation project funds rooftop rainwater systems for peri-urban households facing seasonal storms. Engineers will develop block-level drainage designs using flood recurrence data and community surveys. The program will implement public workshops on safe retrofitting for low-income homes. We will ensure transparent procurement through published contractor performance reports."},
        {"id": "P03", "domain": "climate", "text": "We propose restoring degraded wetlands that buffer river overflow near farming settlements. Community rangers will support biodiversity monitoring with low-cost sensors and mobile logs. The project will provide financing for women-led nurseries to produce native seedlings. District officers will improve maintenance planning through quarterly resilience reviews."},
        # health
        {"id": "P04", "domain": "health", "text": "This health systems initiative upgrades rural clinics with solar backup and vaccine cold-chain equipment. Nurses will implement offline triage tools that synchronize records when connectivity returns. The project will provide transport stipends for community health workers conducting follow-up visits. Supervisors will ensure referral compliance through monthly audit meetings."},
        {"id": "P05", "domain": "health", "text": "Our maternal care proposal creates emergency obstetric simulation labs in district hospitals. Midwives will develop shared response protocols with ambulance teams and referral centers. The program will support hotline routing for high-risk pregnancies in remote villages. Hospital boards will improve accountability by publishing adverse-event response timelines."},
        {"id": "P06", "domain": "health", "text": "We will implement an urban primary care reform that extends clinic hours for informal workers. The project provides digital queueing and appointment reminders in local languages. Community advocates will support enrollment for uninsured households using assisted registration desks. City health units will ensure service quality with patient experience scorecards."},
        # education
        {"id": "P07", "domain": "education", "text": "This education initiative develops open-source lesson authoring tools for low-bandwidth schools. Teachers will provide peer-reviewed content aligned with national curriculum milestones. The project will support formative assessment packs that run offline on shared tablets. District mentors will improve classroom adoption through coaching visits and feedback loops."},
        {"id": "P08", "domain": "education", "text": "Our literacy acceleration program creates community reading clubs in underserved settlements. Facilitators will implement bilingual story modules for early grade learners and caregivers. The project provides micro-grants for schools to develop print-rich classroom corners. Monitoring teams will ensure consistent attendance tracking and remedial follow-up."},
        {"id": "P09", "domain": "education", "text": "We propose vocational bridge courses linking secondary students to local green jobs. Colleges will support short certifications in solar installation, repair, and safety. The project will provide scholarship financing for girls from low-income households. Career offices will improve placement outcomes through employer partnership dashboards."},
        # governance
        {"id": "P10", "domain": "governance", "text": "This civic transparency project implements an open contracting portal for municipal tenders. Citizens will provide feedback on bid irregularities through verified reporting channels. The program will support watchdog groups with legal templates and evidence management workflows. Oversight committees will ensure response timelines are publicly visible."},
        {"id": "P11", "domain": "governance", "text": "Our anti-corruption initiative develops participatory budgeting forums at ward level. Community delegates will implement scorecards to evaluate service delivery against allocated funds. The project provides facilitation training for women and youth representatives. Local councils will improve trust by publishing procurement amendments in plain language."},
        {"id": "P12", "domain": "governance", "text": "We will create a grievance redress platform for public benefit programs. Caseworkers will support residents in filing complaints via SMS and assisted kiosks. The project will provide analytics for agencies to identify repeat failure points. Independent reviewers will ensure fair resolution through random case quality checks."},
        # agriculture
        {"id": "P13", "domain": "agriculture", "text": "This agriculture project implements shared soil testing services for smallholder cooperatives. Extension officers will provide crop recommendations based on rainfall and nutrient diagnostics. The program will support women farmers with financing for drip irrigation upgrades. Producer groups will ensure transparent input purchasing through digital ledgers."},
        {"id": "P14", "domain": "agriculture", "text": "Our farm resilience initiative develops pest early-warning alerts using village scouting networks. Agronomists will implement demonstration plots for low-chemical integrated pest management. The project provides seed banks that support climate-tolerant varieties for staple crops. Cooperative leaders will improve planning by reviewing yield data each season."},
        {"id": "P15", "domain": "agriculture", "text": "We propose post-harvest storage hubs to reduce losses in remote farming corridors. The project will provide cooling units and moisture monitoring equipment at collection points. Farmer associations will support quality grading and contract negotiation with buyers. District planners will ensure route maintenance for reliable market access."},
        # water
        {"id": "P16", "domain": "water", "text": "This water access initiative rehabilitates boreholes in drought-prone villages. Committees will implement preventive maintenance schedules and publish tariff decisions monthly. The project will provide chlorine dosing kiosks near schools and health posts. Sensor alerts will ensure rapid repair dispatch when pump faults occur."},
        {"id": "P17", "domain": "water", "text": "Our sanitation proposal develops faecal sludge transfer systems for dense informal settlements. Operators will support safe emptying standards and protective equipment compliance. The project provides financing for neighborhood transfer stations linked to treatment facilities. Municipal teams will improve oversight through digital route and disposal logs."},
        {"id": "P18", "domain": "water", "text": "We will implement watershed restoration in upper catchment communities. The project provides terracing and vegetation bundles to reduce sediment runoff into reservoirs. Community stewards will support stream monitoring with simple turbidity kits. Water utilities will ensure shared governance through annual source protection agreements."},
        # legal
        {"id": "P19", "domain": "legal", "text": "This legal aid initiative creates mobile clinics for displaced families in border districts. Paralegals will provide document preparation support for asylum and status regularization. The project will implement multilingual hearing reminders through SMS and hotline channels. Partner firms will ensure representation for complex protection appeals."},
        {"id": "P20", "domain": "legal", "text": "Our justice access project develops community mediation hubs for low-income neighborhoods. Trained facilitators will support dispute resolution before cases escalate to courts. The program provides legal literacy sessions focused on tenancy and labor protections. Case managers will improve follow-up with secure digital case notes."},
        {"id": "P21", "domain": "legal", "text": "We propose a women-focused legal defense fund for survivors of economic abuse. The project will provide emergency filing support and temporary income protection referrals. Community advocates will implement rights education workshops in local languages. Monitoring panels will ensure confidentiality safeguards across all partner services."},
        # open science
        {"id": "P22", "domain": "open_science", "text": "This open science proposal develops a federated data commons for environmental research. Institutions will provide standardized metadata and reproducible analysis notebooks. The project will support shared licensing templates for civic and academic reuse. Stewardship boards will ensure equitable access through transparent review criteria."},
        {"id": "P23", "domain": "open_science", "text": "Our reproducibility initiative implements public protocol registries for health intervention studies. Research teams will support versioned datasets and pre-analysis plans before data collection. The project provides training for early-career scientists on open methods and ethical governance. University alliances will improve compliance with quarterly peer audits."},
        {"id": "P24", "domain": "open_science", "text": "We will create a citizen science platform for biodiversity observations in peri-urban ecosystems. Volunteers can provide geotagged records using offline-capable mobile tools. The project will implement validation workflows led by local universities and conservation groups. Open dashboards will ensure that findings inform municipal land-use debates."},
        # urban infrastructure
        {"id": "P25", "domain": "urban_infrastructure", "text": "This urban infrastructure initiative implements bus-priority corridors on congested commuter routes. Transport agencies will provide open timetable and crowding data for route optimization. The project supports station upgrades with universal accessibility features and safe lighting. Rider councils will ensure accountability through weekly service scorecards."},
        {"id": "P26", "domain": "urban_infrastructure", "text": "Our settlement upgrading proposal develops flood-resilient walkways and drainage in informal neighborhoods. Engineers will support participatory mapping to prioritize high-risk blocks. The project provides financing for local contractors trained in inclusive design standards. Ward teams will improve maintenance with resident reporting channels."},
        {"id": "P27", "domain": "urban_infrastructure", "text": "We propose community-managed mini-grids for markets and transit nodes with unreliable electricity. The project will provide smart meters and transparent billing dashboards for users. Technical teams will implement routine inspections and safety drills with local operators. Municipal partners will ensure governance through public tariff consultations."},
        # digital inclusion
        {"id": "P28", "domain": "digital_inclusion", "text": "This digital inclusion project creates community device labs in public libraries. Mentors will provide foundational courses on online safety and productivity tools. The initiative will implement targeted sessions for older adults and first-time internet users. Library networks will ensure continued support through volunteer help desks."},
        {"id": "P29", "domain": "digital_inclusion", "text": "Our connectivity proposal develops neighborhood Wi-Fi cooperatives in underserved urban peripheries. Resident committees will support fair usage policies and maintenance contributions. The project provides financing for shared backhaul links and resilient backup power. Local schools will improve learning continuity through subsidized student access plans."},
        {"id": "P30", "domain": "digital_inclusion", "text": "We will implement an assistive technology access program for learners with disabilities. Schools can provide screen readers, captioning tools, and adaptable input devices. The project supports teacher training on inclusive digital pedagogy and classroom accommodations. Parent associations will ensure feedback loops for continuous service improvement."},
    ]


def build_rule_variants(base: List[dict]) -> Dict[str, dict]:
    variants: Dict[str, dict] = {}
    for item in base:
        pid = item["id"]
        text = item["text"]
        variants[f"{pid}_rule_shuffle"] = {"base_id": pid, "type": "rule", "variant": "shuffle", "text": shuffle_variant(text)}
        variants[f"{pid}_rule_synonym"] = {"base_id": pid, "type": "rule", "variant": "synonym", "text": synonym_variant(text)}
        variants[f"{pid}_rule_passive"] = {"base_id": pid, "type": "rule", "variant": "passive", "text": passive_variant(text)}
    return variants


def build_claude_variants(base: List[dict], client: AgentRouterClient, paraphrase_cache: Dict[str, dict]) -> Dict[str, dict]:
    variants: Dict[str, dict] = {}
    for item in base:
        pid = item["id"]
        text = item["text"]
        if pid not in paraphrase_cache:
            print(f"Generating Claude paraphrases for {pid}")
            try:
                paraphrase_cache[pid] = client.generate_paraphrases(text)
                save_json(PARAPHRASE_CACHE_PATH, paraphrase_cache)
            except Exception as exc:
                print(f"Paraphrase failed for {pid}: {exc}")
                continue

        bundle = paraphrase_cache.get(pid, {})
        for key in ["formal", "concise", "narrative"]:
            value = bundle.get(key)
            if value:
                variants[f"{pid}_claude_{key}"] = {
                    "base_id": pid,
                    "type": "claude",
                    "variant": key,
                    "text": value,
                }
    return variants


def build_pairs(base: List[dict], rule_variants: Dict[str, dict], claude_variants: Dict[str, dict]) -> List[dict]:
    by_id = {x["id"]: x for x in base}
    pairs: List[dict] = []

    for vid, v in rule_variants.items():
        base_id = v["base_id"]
        pairs.append(
            {
                "base_id": base_id,
                "variant_id": vid,
                "text_a": by_id[base_id]["text"],
                "text_b": v["text"],
                "label": "duplicate",
                "type": "rule",
            }
        )

    for vid, v in claude_variants.items():
        base_id = v["base_id"]
        pairs.append(
            {
                "base_id": base_id,
                "variant_id": vid,
                "text_a": by_id[base_id]["text"],
                "text_b": v["text"],
                "label": "duplicate",
                "type": "claude",
            }
        )

    domains: Dict[str, List[str]] = {}
    for item in base:
        domains.setdefault(item["domain"], []).append(item["id"])
    for domain in domains:
        domains[domain] = sorted(domains[domain])

    for domain, ids in domains.items():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                pairs.append(
                    {
                        "base_id": a,
                        "variant_id": b,
                        "text_a": by_id[a]["text"],
                        "text_b": by_id[b]["text"],
                        "label": "not_duplicate",
                        "type": "hard_negative",
                    }
                )

    firsts = [ids[0] for ids in domains.values()]
    for i in range(len(firsts)):
        for j in range(i + 1, len(firsts)):
            a, b = firsts[i], firsts[j]
            pairs.append(
                {
                    "base_id": a,
                    "variant_id": b,
                    "text_a": by_id[a]["text"],
                    "text_b": by_id[b]["text"],
                    "label": "not_duplicate",
                    "type": "hard_negative",
                }
            )
    return pairs


def ensure_embeddings(
    pairs: List[dict],
    client: AgentRouterClient,
    embed_cache: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    texts: Dict[str, str] = {}
    for p in pairs:
        aid = p["base_id"]
        bid = p["variant_id"]
        texts[aid] = p["text_a"]
        texts[bid] = p["text_b"]

    done = 0
    total = len(texts)
    for tid, txt in texts.items():
        done += 1
        if tid in embed_cache:
            continue
        print(f"Embedding {done}/{total}: {tid}")
        try:
            embed_cache[tid] = client.get_embedding(txt, EMBEDDING_MODEL)
            save_json(EMBED_CACHE_PATH, embed_cache)
        except Exception as exc:
            print(f"Embedding failed for {tid}: {exc}")
    return embed_cache


def add_scores(pairs: List[dict], embed_cache: Dict[str, List[float]]) -> None:
    for p in pairs:
        va = embed_cache.get(p["base_id"])
        vb = embed_cache.get(p["variant_id"])
        if va is None or vb is None:
            p["similarity_score"] = None
            p["predictions"] = {f"{t:.2f}": None for t in THRESHOLDS}
            continue
        sim = cosine_sim(va, vb)
        p["similarity_score"] = sim
        p["predictions"] = {f"{t:.2f}": ("duplicate" if sim >= t else "not_duplicate") for t in THRESHOLDS}


def prf(positives: List[dict], negatives: List[dict], threshold: float) -> Tuple[float, float, float]:
    tp = fp = fn = 0
    for p in positives:
        s = p.get("similarity_score")
        if s is None:
            continue
        pred_dup = s >= threshold
        if pred_dup:
            tp += 1
        else:
            fn += 1
    for n in negatives:
        s = n.get("similarity_score")
        if s is None:
            continue
        if s >= threshold:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def dist(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    client = EvaluationClient()
    paraphrase_cache = load_json(PARAPHRASE_CACHE_PATH)
    embed_cache = load_json(EMBED_CACHE_PATH)

    base = build_base_proposals()
    rule_variants = build_rule_variants(base)
    claude_variants = build_claude_variants(base, client, paraphrase_cache)
    pairs = build_pairs(base, rule_variants, claude_variants)

    print(f"Pairs prepared: {len(pairs)}")
    print("Fetching embeddings...")
    embed_cache = ensure_embeddings(pairs, client, embed_cache)
    add_scores(pairs, embed_cache)

    required_text_ids = {p["base_id"] for p in pairs} | {p["variant_id"] for p in pairs}
    available_text_ids = set(embed_cache.keys())
    missing_ids = sorted(required_text_ids - available_text_ids)
    if missing_ids:
        sample = ", ".join(missing_ids[:10])
        suffix = " ..." if len(missing_ids) > 10 else ""
        raise RuntimeError(
            "Insufficient embeddings for duplicate evaluation. "
            f"Missing {len(missing_ids)} text embeddings (e.g., {sample}{suffix}). "
            "Check network/DNS access and GEMINI_API_KEY or ANTHROPIC_API_KEY configuration."
        )

    negatives = [p for p in pairs if p["label"] == "not_duplicate"]
    rule_pos = [p for p in pairs if p["label"] == "duplicate" and p["type"] == "rule"]
    claude_pos = [p for p in pairs if p["label"] == "duplicate" and p["type"] == "claude"]
    combined_pos = [p for p in pairs if p["label"] == "duplicate"]

    scored_negatives = [p for p in negatives if p.get("similarity_score") is not None]
    scored_rule_pos = [p for p in rule_pos if p.get("similarity_score") is not None]
    scored_claude_pos = [p for p in claude_pos if p.get("similarity_score") is not None]
    if not scored_negatives or not scored_rule_pos or not scored_claude_pos:
        raise RuntimeError(
            "Insufficient scored pairs for duplicate metrics. "
            f"rule={len(scored_rule_pos)}, claude={len(scored_claude_pos)}, negatives={len(scored_negatives)}. "
            "Evaluation aborted to avoid misleading metrics."
        )

    threshold_rows = []
    best = None
    for th in THRESHOLDS:
        r_p, r_r, r_f = prf(rule_pos, negatives, th)
        c_p, c_r, c_f = prf(claude_pos, negatives, th)
        m_p, m_r, m_f = prf(combined_pos, negatives, th)
        threshold_rows.append(
            {
                "threshold": th,
                "rule_precision": r_p,
                "rule_recall": r_r,
                "rule_f1": r_f,
                "claude_precision": c_p,
                "claude_recall": c_r,
                "claude_f1": c_f,
                "combined_precision": m_p,
                "combined_recall": m_r,
                "combined_f1": m_f,
            }
        )
        if best is None or (m_f > best["combined_f1"]) or (abs(m_f - best["combined_f1"]) < 1e-12 and th > best["threshold"]):
            best = threshold_rows[-1]

    rule_scores = [p["similarity_score"] for p in rule_pos if p.get("similarity_score") is not None]
    claude_scores = [p["similarity_score"] for p in claude_pos if p.get("similarity_score") is not None]
    neg_scores = [p["similarity_score"] for p in negatives if p.get("similarity_score") is not None]
    d_rule = dist(rule_scores)
    d_claude = dist(claude_scores)
    d_neg = dist(neg_scores)

    save_json(
        RESULT_JSON_PATH,
        {
            "pairs": pairs,
            "score_distribution": {
                "rule_duplicates": d_rule,
                "claude_duplicates": d_claude,
                "hard_negatives": d_neg,
            },
            "threshold_metrics": threshold_rows,
            "recommended_threshold": best,
        },
    )

    dist_table = tabulate(
        [
            ["Rule duplicates", f"{d_rule['mean']:.4f}", f"{d_rule['std']:.4f}", f"{d_rule['min']:.4f}", f"{d_rule['max']:.4f}"],
            ["Claude duplicates", f"{d_claude['mean']:.4f}", f"{d_claude['std']:.4f}", f"{d_claude['min']:.4f}", f"{d_claude['max']:.4f}"],
            ["Hard negatives", f"{d_neg['mean']:.4f}", f"{d_neg['std']:.4f}", f"{d_neg['min']:.4f}", f"{d_neg['max']:.4f}"],
        ],
        headers=["Subset", "Mean", "Std", "Min", "Max"],
        tablefmt="github",
    )

    metric_table = tabulate(
        [
            [
                f"{r['threshold']:.2f}",
                f"{r['rule_precision']:.4f}",
                f"{r['rule_recall']:.4f}",
                f"{r['rule_f1']:.4f}",
                f"{r['claude_precision']:.4f}",
                f"{r['claude_recall']:.4f}",
                f"{r['claude_f1']:.4f}",
                f"{r['combined_precision']:.4f}",
                f"{r['combined_recall']:.4f}",
                f"{r['combined_f1']:.4f}",
            ]
            for r in threshold_rows
        ],
        headers=[
            "Threshold",
            "Rule P",
            "Rule R",
            "Rule F1",
            "Claude P",
            "Claude R",
            "Claude F1",
            "Combined P",
            "Combined R",
            "Combined F1",
        ],
        tablefmt="github",
    )

    report_lines = [
        "# Duplicate Detection Evaluation",
        "",
        "## Score distribution",
        "",
        dist_table,
        "",
        "## Precision/Recall/F1 by threshold",
        "",
        metric_table,
        "",
        "## Recommended threshold",
        "",
        f"`DUPLICATE_THRESHOLD={best['threshold']:.2f}`",
        "",
        f"Selected because it achieves the highest combined F1 ({best['combined_f1']:.4f}); ties are resolved toward a higher threshold to reduce false positives.",
    ]
    with REPORT_MD_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n=== Duplicate Evaluation Complete ===")
    print(f"Recommended threshold: DUPLICATE_THRESHOLD={best['threshold']:.2f}")
    print(f"Combined F1 at recommended threshold: {best['combined_f1']:.4f}")


if __name__ == "__main__":
    main()
