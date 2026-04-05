#!/usr/bin/env python3
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine
from tabulate import tabulate
from urllib import error, request


RESULTS_DIR = Path(__file__).resolve().parent / "results"
CACHE_PATH = RESULTS_DIR / "embedding_cache.json"
RESULT_JSON_PATH = RESULTS_DIR / "multilingual_results.json"
REPORT_MD_PATH = RESULTS_DIR / "multilingual_report.md"

EMBEDDING_MODEL = "text-embedding-004"
MAX_RETRIES = 5
INITIAL_BACKOFF = 2


@dataclass
class PairScore:
    proposal_id: int
    language_pair: str
    similarity: float
    label: str


class AgentRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = os.getenv("AGENTROUTER_BASE_URL", "https://api.agentrouter.ai/v1").rstrip("/")
        self._sdk_client = None
        try:
            from agentrouter import AgentRouter  # type: ignore

            self._sdk_client = AgentRouter(api_key=api_key)
        except Exception:
            self._sdk_client = None

    def _http_post(self, endpoint: str, payload: dict) -> dict:
        req = request.Request(
            f"{self.base_url}/{endpoint.lstrip('/')}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _with_backoff(self, fn):
        backoff = INITIAL_BACKOFF
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return fn()
            except Exception as exc:
                msg = str(exc).lower()
                is_rate_limited = "429" in msg or "rate" in msg
                if not is_rate_limited or attempt == MAX_RETRIES:
                    raise
                print(f"Rate limit detected, retrying in {backoff}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError("Retry loop exhausted unexpectedly")

    def get_embedding(self, text: str, model: str) -> List[float]:
        def call():
            if self._sdk_client is not None:
                if hasattr(self._sdk_client, "embeddings") and hasattr(self._sdk_client.embeddings, "create"):
                    resp = self._sdk_client.embeddings.create(model=model, input=text)
                    data = getattr(resp, "data", None) or resp.get("data", [])
                    if data:
                        first = data[0]
                        emb = getattr(first, "embedding", None) or first.get("embedding")
                        if emb:
                            return emb
                if hasattr(self._sdk_client, "responses") and hasattr(self._sdk_client.responses, "create"):
                    resp = self._sdk_client.responses.create(model=model, input=text)
                    output = getattr(resp, "output", None) or resp.get("output", [])
                    for item in output:
                        content = getattr(item, "content", None) or item.get("content", [])
                        for part in content:
                            emb = getattr(part, "embedding", None) or part.get("embedding")
                            if emb:
                                return emb

            data = self._http_post("embeddings", {"model": model, "input": text})
            if "data" in data and data["data"]:
                return data["data"][0]["embedding"]
            if "embedding" in data:
                return data["embedding"]
            raise RuntimeError(f"Unexpected embedding response shape: {data}")

        return self._with_backoff(call)


def safe_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(1 - cosine(a, b))


def load_cache(path: Path) -> Dict[str, List[float]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(path: Path, cache: Dict[str, List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f)


def get_proposals() -> Dict[int, Dict[str, str]]:
    return {
        1: {
            "en": "Our climate adaptation program will restore coastal mangroves and train local youth as ecosystem stewards. Municipal planners will receive open hazard maps that combine flood history and elevation data. We will pilot household rainwater retention kits in three vulnerable wards. The goal is to reduce seasonal flood losses while creating green jobs.",
            "es": "Nuestro programa de adaptación climática restaurará manglares costeros y capacitará a jóvenes locales como guardianes del ecosistema. Los planificadores municipales recibirán mapas abiertos de riesgo que combinan historial de inundaciones y datos de elevación. Pondremos a prueba kits de retención de agua de lluvia para hogares en tres barrios vulnerables. El objetivo es reducir las pérdidas por inundaciones estacionales mientras se crean empleos verdes.",
            "sw": "Mpango wetu wa kukabiliana na mabadiliko ya tabianchi utarejesha mikoko ya pwani na kuwafundisha vijana wa eneo kuwa walinzi wa mazingira. Wapangaji wa manispaa watapokea ramani huria za hatari zinazounganisha historia ya mafuriko na data ya mwinuko. Tutajaribu vifaa vya kuhifadhi maji ya mvua kwa kaya katika kata tatu zilizo hatarini. Lengo ni kupunguza hasara za mafuriko ya msimu huku tukitengeneza ajira za kijani.",
            "pt": "Nosso programa de adaptação climática vai restaurar manguezais costeiros e treinar jovens locais como guardiões do ecossistema. Planejadores municipais receberão mapas abertos de risco que combinam histórico de enchentes e dados de elevação. Vamos pilotar kits domiciliares de retenção de água da chuva em três bairros vulneráveis. O objetivo é reduzir perdas com enchentes sazonais enquanto criamos empregos verdes.",
        },
        2: {
            "en": "This rural health infrastructure project upgrades six clinics with solar power, vaccine refrigeration, and telemedicine corners. Nurses will use an offline-first patient record tool that syncs when connectivity returns. Community health workers will receive motorbike transport stipends for outreach. The intervention is designed to reduce missed maternal and child appointments.",
            "es": "Este proyecto de infraestructura de salud rural moderniza seis clínicas con energía solar, refrigeración de vacunas y espacios de telemedicina. Las enfermeras usarán una herramienta de historiales clínicos que funciona sin conexión y sincroniza cuando regresa la conectividad. Los trabajadores comunitarios de salud recibirán estipendios de transporte en motocicleta para visitas de alcance. La intervención está diseñada para reducir las citas maternas e infantiles perdidas.",
            "sw": "Mradi huu wa miundombinu ya afya vijijini unaboresha kliniki sita kwa nishati ya jua, uhifadhi wa chanjo kwenye jokofu, na maeneo ya telemedicine. Wauguzi watatumia kifaa cha rekodi za wagonjwa kinachofanya kazi bila mtandao na kusawazisha data mtandao ukirudi. Wahudumu wa afya ya jamii watapokea posho za usafiri wa pikipiki kwa huduma za uhamasishaji. Hatua hii imeundwa kupunguza miadi ya mama na mtoto inayokosekana.",
            "pt": "Este projeto de infraestrutura de saúde rural moderniza seis clínicas com energia solar, refrigeração de vacinas e espaços de telemedicina. Enfermeiras usarão uma ferramenta de prontuário que funciona offline e sincroniza quando a conectividade retorna. Agentes comunitários de saúde receberão bolsas para transporte por motocicleta em ações de alcance. A intervenção foi desenhada para reduzir faltas em consultas maternas e infantis.",
        },
        3: {
            "en": "We will build open-source education tooling for low-bandwidth classrooms in public schools. The platform includes printable lesson packs, lightweight assessments, and teacher analytics dashboards. Local teacher fellows will co-design content in math and science aligned to national standards. District supervisors will monitor learning recovery using transparent indicators.",
            "es": "Construiremos herramientas educativas de código abierto para aulas de baja conectividad en escuelas públicas. La plataforma incluye paquetes de lecciones imprimibles, evaluaciones ligeras y paneles analíticos para docentes. Docentes becarios locales codiseñarán contenidos de matemáticas y ciencias alineados con estándares nacionales. Los supervisores distritales monitorearán la recuperación del aprendizaje con indicadores transparentes.",
            "sw": "Tutaunda zana huria za elimu kwa madarasa yenye intaneti dhaifu katika shule za umma. Mfumo utajumuisha vifurushi vya masomo vinavyoweza kuchapishwa, tathmini nyepesi, na dashibodi za uchambuzi kwa walimu. Walimu wenzetu wa eneo watashirikiana kubuni maudhui ya hisabati na sayansi yanayolingana na viwango vya taifa. Wasimamizi wa wilaya watafuatilia urejeshaji wa ujifunzaji kwa viashiria vya uwazi.",
            "pt": "Vamos construir ferramentas educacionais de código aberto para salas de aula de baixa conectividade em escolas públicas. A plataforma inclui pacotes de aula imprimíveis, avaliações leves e painéis analíticos para professores. Professores bolsistas locais cocriarão conteúdos de matemática e ciências alinhados aos padrões nacionais. Supervisores distritais acompanharão a recuperação da aprendizagem com indicadores transparentes.",
        },
        4: {
            "en": "Our anti-corruption civic tech proposal launches a public procurement observatory for municipal contracts. Citizens can track tender timelines, contract changes, and vendor ownership in one interface. Civil society monitors will be trained to flag red-flag patterns and submit evidence packets to oversight bodies. The project intends to increase accountability and reduce procurement leakage.",
            "es": "Nuestra propuesta de tecnología cívica anticorrupción lanza un observatorio de compras públicas para contratos municipales. La ciudadanía podrá seguir cronogramas de licitación, cambios contractuales y propiedad de proveedores en una sola interfaz. Monitores de la sociedad civil serán capacitados para señalar patrones de alerta y enviar expedientes de evidencia a órganos de control. El proyecto busca aumentar la rendición de cuentas y reducir fugas en las compras.",
            "sw": "Pendekezo letu la teknolojia ya kiraia dhidi ya ufisadi linazindua kituo cha ufuatiliaji wa ununuzi wa umma kwa mikataba ya manispaa. Wananchi wataweza kufuatilia ratiba za zabuni, mabadiliko ya mkataba, na umiliki wa wauzaji katika kiolesura kimoja. Waangalizi wa asasi za kiraia watafunzwa kutambua viashiria vya hatari na kuwasilisha vifurushi vya ushahidi kwa taasisi za usimamizi. Mradi unalenga kuongeza uwajibikaji na kupunguza upotevu katika ununuzi.",
            "pt": "Nossa proposta de tecnologia cívica anticorrupção lança um observatório de compras públicas para contratos municipais. Cidadãos poderão acompanhar cronogramas de licitação, alterações contratuais e propriedade de fornecedores em uma única interface. Monitores da sociedade civil serão treinados para identificar padrões de risco e enviar dossiês de evidências aos órgãos de controle. O projeto pretende ampliar a responsabilização e reduzir perdas em compras públicas.",
        },
        5: {
            "en": "This smallholder agriculture initiative will establish shared soil testing hubs and climate-smart extension services. Farmer groups will receive SMS advisories tailored to local weather and market prices. Women-led cooperatives will access revolving micro-grants for irrigation upgrades. The approach aims to raise yields while reducing input waste.",
            "es": "Esta iniciativa de agricultura de pequeños productores establecerá centros compartidos de análisis de suelos y servicios de extensión climáticamente inteligentes. Los grupos de agricultores recibirán avisos por SMS adaptados al clima local y a precios de mercado. Cooperativas lideradas por mujeres accederán a microsubvenciones rotativas para mejorar el riego. El enfoque busca aumentar rendimientos mientras reduce el desperdicio de insumos.",
            "sw": "Mpango huu wa kilimo kwa wakulima wadogo utaanzisha vituo vya pamoja vya upimaji wa udongo na huduma za ugani zinazozingatia tabianchi. Vikundi vya wakulima vitapokea ushauri kwa SMS unaolingana na hali ya hewa ya eneo na bei za soko. Vyama vya ushirika vinavyoongozwa na wanawake vitapata ruzuku ndogo za mzunguko kwa uboreshaji wa umwagiliaji. Mbinu hii inalenga kuongeza mavuno huku ikipunguza upotevu wa pembejeo.",
            "pt": "Esta iniciativa de agricultura familiar estabelecerá centros compartilhados de análise de solo e serviços de extensão climática inteligente. Grupos de agricultores receberão avisos por SMS adaptados ao clima local e aos preços de mercado. Cooperativas lideradas por mulheres terão acesso a microbolsas rotativas para modernizar a irrigação. A abordagem busca elevar a produtividade enquanto reduz desperdícios de insumos.",
        },
        6: {
            "en": "We propose a clean water access program that rehabilitates boreholes and installs chlorine dosing kiosks. Village water committees will publish maintenance logs and tariff decisions monthly. Sensor alerts will notify mechanics when pumps fail to reduce downtime. The project prioritizes schools and clinics in drought-prone settlements.",
            "es": "Proponemos un programa de acceso a agua limpia que rehabilita pozos e instala quioscos de dosificación de cloro. Los comités de agua comunitarios publicarán mensualmente registros de mantenimiento y decisiones tarifarias. Alertas de sensores notificarán a mecánicos cuando fallen las bombas para reducir el tiempo fuera de servicio. El proyecto prioriza escuelas y clínicas en asentamientos propensos a la sequía.",
            "sw": "Tunapendekeza mpango wa upatikanaji wa maji safi unaokarabati visima na kusakinisha vibanda vya kuongeza klorini. Kamati za maji vijijini zitachapisha kila mwezi kumbukumbu za matengenezo na maamuzi ya tozo. Arifa za vihisi zitawajulisha mafundi pampu zinapoharibika ili kupunguza muda wa kukatika huduma. Mradi unatanguliza shule na kliniki katika makazi yanayokumbwa na ukame.",
            "pt": "Propomos um programa de acesso à água limpa que reabilita poços e instala quiosques de dosagem de cloro. Comitês de água comunitários publicarão mensalmente registros de manutenção e decisões tarifárias. Alertas de sensores avisarão mecânicos quando bombas falharem para reduzir tempo de inatividade. O projeto prioriza escolas e clínicas em assentamentos sujeitos à seca.",
        },
        7: {
            "en": "This legal aid for refugees program creates mobile case clinics at border districts and urban reception centers. Paralegals will help families file asylum documents and track hearing dates through a multilingual SMS system. Partner law firms will provide pro bono representation for complex protection claims. Our objective is to reduce case abandonment and improve due-process outcomes.",
            "es": "Este programa de asistencia legal para personas refugiadas crea clínicas móviles de casos en distritos fronterizos y centros urbanos de recepción. Paralegales ayudarán a las familias a presentar documentos de asilo y seguir fechas de audiencia mediante un sistema SMS multilingüe. Firmas legales asociadas brindarán representación pro bono para casos complejos de protección. Nuestro objetivo es reducir el abandono de casos y mejorar resultados de debido proceso.",
            "sw": "Mpango huu wa msaada wa kisheria kwa wakimbizi unaanzisha kliniki za kesi zinazohama katika wilaya za mipakani na vituo vya mapokezi mijini. Wasaidizi wa sheria watasaidia familia kuwasilisha nyaraka za hifadhi na kufuatilia tarehe za usikilizwaji kupitia mfumo wa SMS wa lugha nyingi. Kampuni washirika za sheria zitatoa uwakilishi wa pro bono kwa madai magumu ya ulinzi. Lengo letu ni kupunguza kuachwa kwa kesi na kuboresha matokeo ya haki ya mchakato.",
            "pt": "Este programa de assistência jurídica para refugiados cria clínicas móveis de casos em distritos de fronteira e centros urbanos de acolhimento. Paralegais ajudarão famílias a protocolar documentos de asilo e acompanhar datas de audiência por um sistema multilíngue de SMS. Escritórios parceiros oferecerão representação pro bono para pedidos complexos de proteção. Nosso objetivo é reduzir abandono de casos e melhorar resultados de devido processo.",
        },
        8: {
            "en": "Our open science data commons will federate public health and environmental datasets under shared governance rules. Researchers can publish reproducible workflows, metadata, and licensing terms in one portal. Community data stewards will review requests from civic groups and local universities. The initiative lowers barriers to evidence-based policy experimentation.",
            "es": "Nuestro común de datos para ciencia abierta federará conjuntos de datos de salud pública y ambiente bajo reglas compartidas de gobernanza. Investigadores podrán publicar flujos reproducibles, metadatos y términos de licencia en un portal único. Guardianes comunitarios de datos revisarán solicitudes de grupos cívicos y universidades locales. La iniciativa reduce barreras para la experimentación de políticas basadas en evidencia.",
            "sw": "Hazina yetu ya data ya sayansi huria itaunganisha seti za data za afya ya umma na mazingira chini ya kanuni za pamoja za utawala. Watafiti wanaweza kuchapisha mtiririko wa kazi unaorudiwa, metadata, na masharti ya leseni katika lango moja. Walinzi wa data wa jamii watapitia maombi kutoka vikundi vya kiraia na vyuo vikuu vya eneo. Mpango huu unapunguza vikwazo kwa majaribio ya sera yanayotegemea ushahidi.",
            "pt": "Nosso commons de dados em ciência aberta federará conjuntos de dados de saúde pública e meio ambiente sob regras compartilhadas de governança. Pesquisadores poderão publicar fluxos reprodutíveis, metadados e termos de licença em um único portal. Guardiões comunitários de dados revisarão solicitações de grupos cívicos e universidades locais. A iniciativa reduz barreiras para experimentação de políticas baseadas em evidências.",
        },
        9: {
            "en": "This urban mobility proposal deploys integrated bus-priority lanes and open fare data for commuters. Informal transport unions will co-design route rationalization to preserve livelihoods. A rider feedback channel will publish punctuality and crowding indicators weekly. The project seeks shorter commutes and lower transport emissions.",
            "es": "Esta propuesta de movilidad urbana despliega carriles integrados de prioridad para autobuses y datos abiertos de tarifas para usuarios. Sindicatos de transporte informal codiseñarán la reorganización de rutas para preservar medios de vida. Un canal de retroalimentación de pasajeros publicará semanalmente indicadores de puntualidad y hacinamiento. El proyecto busca traslados más cortos y menores emisiones del transporte.",
            "sw": "Pendekezo hili la uhamaji mijini litaanzisha njia jumuishi za kipaumbele kwa mabasi na data wazi za nauli kwa wasafiri. Vyama vya usafiri usio rasmi vitashirikiana kubuni upangaji upya wa njia ili kulinda kipato cha wanachama. Kituo cha maoni ya abiria kitachapisha kila wiki viashiria vya kufika kwa wakati na msongamano. Mradi unalenga safari fupi na kupunguza uzalishaji wa hewa chafu wa usafiri.",
            "pt": "Esta proposta de mobilidade urbana implanta faixas integradas de prioridade para ônibus e dados abertos de tarifas para passageiros. Sindicatos de transporte informal cocriarão a racionalização de rotas para preservar meios de subsistência. Um canal de feedback dos usuários publicará semanalmente indicadores de pontualidade e lotação. O projeto busca viagens mais curtas e menores emissões no transporte.",
        },
        10: {
            "en": "We will launch a biodiversity conservation corridor linking community forests with degraded watershed zones. Indigenous ranger teams will monitor wildlife movement using low-cost acoustic sensors. Benefit-sharing agreements will reward villages for verified habitat restoration. The program protects species while strengthening local stewardship institutions.",
            "es": "Lanzaremos un corredor de conservación de biodiversidad que conecta bosques comunitarios con zonas degradadas de cuenca. Equipos indígenas de guardabosques monitorearán movimiento de fauna con sensores acústicos de bajo costo. Acuerdos de reparto de beneficios recompensarán a aldeas por restauración verificada del hábitat. El programa protege especies mientras fortalece instituciones locales de gestión.",
            "sw": "Tutazindua ukanda wa uhifadhi wa bioanuwai unaounganisha misitu ya jamii na maeneo ya vyanzo vya maji yaliyoharibika. Timu za walinzi wa asili wa kiasili zitafuatilia mienendo ya wanyamapori kwa vihisi vya sauti vya gharama nafuu. Makubaliano ya kugawana manufaa yatazawadia vijiji kwa urejeshaji wa makazi uliothibitishwa. Mpango unalinda spishi huku ukiimarisha taasisi za usimamizi wa jamii.",
            "pt": "Vamos lançar um corredor de conservação da biodiversidade ligando florestas comunitárias a zonas de bacias degradadas. Equipes indígenas de guardas monitorarão o movimento da fauna com sensores acústicos de baixo custo. Acordos de repartição de benefícios recompensarão vilas por restauração de habitat verificada. O programa protege espécies enquanto fortalece instituições locais de gestão.",
        },
        11: {
            "en": "This maternal health project establishes respectful maternity care units in district hospitals. Midwives will receive simulation-based emergency obstetric training and supervision. A transport voucher system will connect high-risk pregnancies to referral centers within two hours. We expect fewer preventable complications and improved patient trust.",
            "es": "Este proyecto de salud materna establece unidades de atención materna respetuosa en hospitales distritales. Las parteras recibirán capacitación obstétrica de emergencia basada en simulación y supervisión continua. Un sistema de vales de transporte conectará embarazos de alto riesgo con centros de referencia en menos de dos horas. Esperamos menos complicaciones prevenibles y mayor confianza de las pacientes.",
            "sw": "Mradi huu wa afya ya uzazi unaanzisha vitengo vya huduma ya heshima kwa wajawazito katika hospitali za wilaya. Wakunga watapokea mafunzo ya dharura za uzazi yanayotumia uigaji na usimamizi endelevu. Mfumo wa vocha za usafiri utaunganisha mimba hatarishi na vituo vya rufaa ndani ya saa mbili. Tunatarajia kupungua kwa matatizo yanayozuilika na kuongezeka kwa imani ya wagonjwa.",
            "pt": "Este projeto de saúde materna estabelece unidades de cuidado materno respeitoso em hospitais distritais. Parteiras receberão treinamento obstétrico de emergência baseado em simulação e supervisão contínua. Um sistema de vouchers de transporte conectará gestações de alto risco a centros de referência em até duas horas. Esperamos menos complicações evitáveis e maior confiança das pacientes.",
        },
        12: {
            "en": "Our digital literacy initiative equips community libraries with device labs and foundational cybersecurity curricula. Youth mentors will run evening classes on online safety, productivity tools, and job search platforms. Parents and older adults will receive tailored modules in local languages. The project closes practical digital skills gaps for excluded households.",
            "es": "Nuestra iniciativa de alfabetización digital equipa bibliotecas comunitarias con laboratorios de dispositivos y currículo básico de ciberseguridad. Mentores juveniles impartirán clases nocturnas sobre seguridad en línea, herramientas de productividad y plataformas de empleo. Madres, padres y personas mayores recibirán módulos adaptados en idiomas locales. El proyecto cierra brechas prácticas de habilidades digitales para hogares excluidos.",
            "sw": "Mpango wetu wa ujuzi wa kidijitali unawezesha maktaba za jamii kwa maabara za vifaa na mtaala wa msingi wa usalama wa mtandao. Washauri vijana wataendesha madarasa ya jioni kuhusu usalama mtandaoni, zana za uzalishaji, na majukwaa ya kutafuta kazi. Wazazi na watu wazima wakubwa watapata moduli zilizobinafsishwa kwa lugha za eneo. Mradi unafunga pengo la ujuzi wa kidijitali kwa kaya zilizotengwa.",
            "pt": "Nossa iniciativa de alfabetização digital equipa bibliotecas comunitárias com laboratórios de dispositivos e currículo básico de cibersegurança. Mentores jovens conduzirão aulas noturnas sobre segurança online, ferramentas de produtividade e plataformas de busca de emprego. Pais e idosos receberão módulos personalizados em línguas locais. O projeto fecha lacunas práticas de habilidades digitais para famílias excluídas.",
        },
    }


def key_for(proposal_id: int, language: str) -> str:
    return f"{proposal_id}:{language}"


def main() -> None:
    api_key = os.getenv("AGENTROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("AGENTROUTER_API_KEY is required")

    np.random.seed(42)

    proposals = get_proposals()
    languages = ["en", "es", "sw", "pt"]
    target_languages = ["es", "sw", "pt"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache(CACHE_PATH)
    client = AgentRouterClient(api_key=api_key)

    embeddings: Dict[Tuple[int, str], List[float]] = {}

    for proposal_id in sorted(proposals):
        for lang in languages:
            cache_key = key_for(proposal_id, lang)
            if cache_key in cache:
                embeddings[(proposal_id, lang)] = cache[cache_key]
                continue

            print(f"Fetching embedding: proposal={proposal_id}, lang={lang}")
            text = proposals[proposal_id][lang]
            try:
                emb = client.get_embedding(text, EMBEDDING_MODEL)
                embeddings[(proposal_id, lang)] = emb
                cache[cache_key] = emb
                save_cache(CACHE_PATH, cache)
            except Exception as exc:
                print(f"Failed embedding for proposal={proposal_id}, lang={lang}: {exc}")

    positive_scores: List[PairScore] = []
    negative_scores: List[PairScore] = []

    for proposal_id in sorted(proposals):
        en_emb = embeddings.get((proposal_id, "en"))
        if en_emb is None:
            continue
        for lang in target_languages:
            other = embeddings.get((proposal_id, lang))
            if other is None:
                continue
            sim = safe_cosine_similarity(en_emb, other)
            positive_scores.append(
                PairScore(
                    proposal_id=proposal_id,
                    language_pair=f"en-{lang}",
                    similarity=sim,
                    label="positive",
                )
            )

    # To satisfy the requested 99 negatives total, build 33 same-language negatives per target language.
    for lang in target_languages:
        for anchor_id in [1, 2, 3]:
            anchor = embeddings.get((anchor_id, lang))
            if anchor is None:
                continue
            for other_id in range(1, 13):
                if other_id == anchor_id:
                    continue
                other = embeddings.get((other_id, lang))
                if other is None:
                    continue
                sim = safe_cosine_similarity(anchor, other)
                negative_scores.append(
                    PairScore(
                        proposal_id=anchor_id,
                        language_pair=f"{lang}-{lang}",
                        similarity=sim,
                        label="negative",
                    )
                )

    pos_vals = np.array([p.similarity for p in positive_scores], dtype=float)
    neg_vals = np.array([p.similarity for p in negative_scores], dtype=float)

    pos_mean = float(np.mean(pos_vals)) if len(pos_vals) else 0.0
    pos_std = float(np.std(pos_vals)) if len(pos_vals) else 0.0
    neg_mean = float(np.mean(neg_vals)) if len(neg_vals) else 0.0
    neg_std = float(np.std(neg_vals)) if len(neg_vals) else 0.0
    gap = pos_mean - neg_mean

    pos_failures = int(np.sum(pos_vals < 0.85)) if len(pos_vals) else 0
    neg_false_positives = int(np.sum(neg_vals >= 0.85)) if len(neg_vals) else 0

    all_pairs = [
        {
            "proposal_id": x.proposal_id,
            "language_pair": x.language_pair,
            "similarity_score": x.similarity,
            "label": x.label,
        }
        for x in (positive_scores + negative_scores)
    ]

    with RESULT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pairs": all_pairs,
                "metrics": {
                    "positive_mean": pos_mean,
                    "positive_std": pos_std,
                    "negative_mean": neg_mean,
                    "negative_std": neg_std,
                    "separation_gap": gap,
                    "positive_failures_below_0_85": pos_failures,
                    "negative_false_positives_above_0_85": neg_false_positives,
                },
            },
            f,
            indent=2,
        )

    table_rows = []
    for pid in range(1, 13):
        row = {"proposal_id": pid, "en-es": None, "en-sw": None, "en-pt": None}
        for p in positive_scores:
            if p.proposal_id == pid:
                row[p.language_pair] = p.similarity
        table_rows.append(
            [
                pid,
                f"{row['en-es']:.4f}" if row["en-es"] is not None else "N/A",
                f"{row['en-sw']:.4f}" if row["en-sw"] is not None else "N/A",
                f"{row['en-pt']:.4f}" if row["en-pt"] is not None else "N/A",
            ]
        )

    neg_by_lang = {"es": [], "sw": [], "pt": []}
    for n in negative_scores:
        lang = n.language_pair.split("-")[0]
        neg_by_lang[lang].append(n.similarity)

    table_rows.append(
        [
            "Negative summary",
            f"{float(np.mean(neg_by_lang['es'])):.4f}" if neg_by_lang["es"] else "N/A",
            f"{float(np.mean(neg_by_lang['sw'])):.4f}" if neg_by_lang["sw"] else "N/A",
            f"{float(np.mean(neg_by_lang['pt'])):.4f}" if neg_by_lang["pt"] else "N/A",
        ]
    )

    table_md = tabulate(table_rows, headers=["Proposal", "EN↔ES", "EN↔SW", "EN↔PT"], tablefmt="github")

    report = [
        "# Multilingual Embedding Evaluation",
        "",
        "## Pairwise Similarity Table",
        "",
        table_md,
        "",
        "## Metrics",
        "",
        f"- Mean positive similarity: `{pos_mean:.4f}`",
        f"- Std positive similarity: `{pos_std:.4f}`",
        f"- Mean negative similarity: `{neg_mean:.4f}`",
        f"- Std negative similarity: `{neg_std:.4f}`",
        f"- Separation gap: `{gap:.4f}`",
        f"- Positive pairs below 0.85: `{pos_failures}`",
        f"- Negative pairs above 0.85: `{neg_false_positives}`",
    ]

    with REPORT_MD_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("\n=== Multilingual Evaluation Summary ===")
    print(f"Positive mean: {pos_mean:.4f} (std={pos_std:.4f})")
    print(f"Negative mean: {neg_mean:.4f} (std={neg_std:.4f})")
    print(f"Separation gap: {gap:.4f}")
    print(f"Positive failures (<0.85): {pos_failures}")
    print(f"Negative false positives (>=0.85): {neg_false_positives}")


if __name__ == "__main__":
    main()
