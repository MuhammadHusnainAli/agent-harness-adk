//! The built-in packs, as shipped with the Python package
//! (`governance/frameworks/packs/*.yaml`), kept here byte for byte so both
//! languages read the same law. Regenerate from the Python files when a pack
//! changes; do not edit by hand.

/// `(id, YAML)`, sorted by id.
pub(crate) const PACKS: &[(&str, &str)] = &[
    (
        "adgm",
        r####"id: adgm
name: ADGM Data Protection Regulations 2021
jurisdiction: ae-adgm
kind: regulation
binding: true
status: in force
as_of: 2026-09-24
summary: GDPR-aligned regime for entities in Abu Dhabi Global Market.
sources:
  - https://www.adgm.com/operating-in-adgm/office-of-data-protection
requirements:
  - {ref: "DPR — principles", title: "Data protection principles", controls: [C5, C7]}
  - {ref: "DPR — rights", title: "Data subject rights including erasure", controls: [C11]}
  - {ref: "DPR — automated decisions", title: "Automated decisions with significant effect", controls: [C8]}
  - {ref: "DPR — breach", title: "Breach notified to the Commissioner within 72 hours", controls: [C15]}
  - {ref: "DPR — DPIA", title: "DPIA for high-risk processing", controls: [C12]}
  - {ref: "DPR — transfers", title: "Transfers outside ADGM", controls: [C6], note: "declare `residency.transfers['ae-adgm']` from the Commissioner's adequacy list"}
  - {ref: "DPR — DPO", title: "Data protection officer where required", manual: true}
incidents:
  personal_data_breach:
    - {notify: ADGM Commissioner of Data Protection, within: 72, basis: "DPR 2021 — 72 hours", min_severity: medium}
"####,
    ),
    (
        "bahrain-pdpl",
        r####"id: bahrain-pdpl
name: Bahrain Personal Data Protection Law (Law No. 30 of 2018)
jurisdiction: bh
kind: law
binding: true
status: in force since 1 August 2019
as_of: 2026-09-24
summary: >-
  Transfers only to countries on the Authority's adequacy list or with its
  authorisation; sensitive data needs explicit consent.
sources:
  - https://www.pdp.gov.bh/en/
requirements:
  - {ref: "Law — principles", title: "Fair and lawful processing for a specified purpose", controls: [C7]}
  - {ref: "Law — sensitive data", title: "Sensitive personal data only with explicit consent", controls: [C4, C7]}
  - {ref: "Law — security", title: "Security of processing", controls: [C5, C10]}
  - {ref: "Art 12-13", title: "Transfers only to adequate or authorised destinations", controls: [C6], note: "add destinations from the Authority's list to `residency.transfers['bh']`"}
  - {ref: "Law — rights", title: "Rights of access, objection and erasure", controls: [C11]}
  - {ref: "Law — automated decisions", title: "Right not to be subject to solely automated decisions", controls: [C8]}
  - {ref: "Law — prior notification", title: "Notification to the Authority before certain processing", manual: true}
policy:
  residency:
    transfers:
      bh: [bh]
    applies_to: [personal]
"####,
    ),
    (
        "ccpa-admt",
        r####"id: ccpa-admt
name: California CCPA regulations on automated decision-making technology and risk assessments
jurisdiction: us
kind: regulation
binding: true
status: CPPA regulations approved 2025; ADMT obligations apply from 1 January 2027
as_of: 2026-09-24
summary: >-
  A business using ADMT for significant decisions gives a pre-use notice, offers an
  opt-out or a human appeal, answers access requests about the ADMT, and performs risk
  assessments for significant-risk processing.
sources:
  - https://cppa.ca.gov/regulations/
requirements:
  - {ref: "Pre-use notice", title: "Notice before ADMT is used for a significant decision", controls: [C9]}
  - {ref: "Opt-out / appeal", title: "Opt-out, or a human appeal of the decision", controls: [C8]}
  - {ref: "Access", title: "Consumers can learn how ADMT was used about them", controls: [C11, C10]}
  - {ref: "Risk assessment", title: "Risk assessment for significant-risk processing", controls: [C12]}
  - {ref: "Deletion", title: "Right to delete", controls: [C11]}
  - {ref: "Cybersecurity audit", title: "Annual cybersecurity audit where thresholds are met", manual: true}
policy:
  rules:
    - id: significant-decisions-human-appeal
      on: tool
      match: {tags: [consequential], jurisdiction: [us-ca]}
      effect: require_approval
      reason: "CCPA ADMT: significant decisions offer human review"
      controls: [C8]
"####,
    ),
    (
        "china-genai",
        r####"id: china-genai
name: China — PIPL, Generative AI Interim Measures, AI Content Labeling Measures and GB 45438-2025
jurisdiction: cn
kind: regulation
binding: true
status: >-
  PIPL in force since 2021; GenAI Interim Measures since 15 Aug 2023; Labeling Measures
  and mandatory standard GB 45438-2025 since 1 Sep 2025
as_of: 2026-09-24
summary: >-
  Generated content carries an explicit label and implicit metadata (provider, content
  id, timestamp); personal information leaves China only after a security assessment,
  standard contract or certification; public-facing GenAI services are filed with the
  CAC. Fines up to RMB 15 million or 5% of turnover.
sources:
  - https://www.chinalawtranslate.com/en/ai-labeling/
  - https://www.twobirds.com/en/insights/2025/new-ai-content-labelling-rules-in-china-what-are-they-and-how-do-they-compare-to-the-eu-ai-act
requirements:
  - {ref: "Labeling Measures + GB 45438", title: "Explicit label on generated text", controls: [C9]}
  - {ref: "Labeling Measures + GB 45438", title: "Implicit label — metadata with provider and content id", controls: [C9]}
  - {ref: "GenAI Measures Art 4", title: "Generated content respects the content rules", manual: true, note: "add an LLMGuard with your content policy"}
  - {ref: "GenAI Measures Art 7", title: "Lawful training and input data", manual: true}
  - {ref: "GenAI Measures Art 17", title: "Security assessment and algorithm filing for public-opinion services", manual: true}
  - {ref: "PIPL Art 28-29", title: "Sensitive personal information needs separate consent", controls: [C4, C7]}
  - {ref: "PIPL Art 38-40", title: "Cross-border transfer only via assessment, standard contract or certification", controls: [C6]}
  - {ref: "PIPL Art 55-56", title: "Personal information protection impact assessment", controls: [C12]}
  - {ref: "PIPL Art 57", title: "Breach remedied and reported", controls: [C15]}
  - {ref: "PIPL Art 24", title: "Automated decisions — transparency and the right to refuse", controls: [C8, C9]}
policy:
  residency:
    transfers:
      cn: [cn]
    applies_to: [personal]
  transparency: {disclose: true, label_output: true, label_for: [cn], provenance: true, languages: [zh]}
incidents:
  personal_data_breach:
    - {notify: personal information protection authority and individuals, within: 0, basis: "PIPL Art 57 — immediately", min_severity: medium}
  security_incident:
    - {notify: CAC / sector regulator, within: 1, basis: "Measures for Reporting Cybersecurity Incidents (2025) — 1 hour for relatively major incidents", min_severity: high}
"####,
    ),
    (
        "colorado-adm",
        r####"id: colorado-adm
name: Colorado automated decision-making law (SB 24-205 as replaced by SB 26-189)
jurisdiction: us
kind: law
binding: true
status: >-
  SB 26-189 signed 14 May 2026; effective 1 January 2027; narrower than SB 24-205 and
  focused on transparency for automated decisions
as_of: 2026-09-24
summary: >-
  Transparency when an automated system makes, or is a substantial factor in, a
  consequential decision about a consumer — education, employment, lending, essential
  government services, health care, housing, insurance, legal services.
sources:
  - https://www.skadden.com/insights/publications/2026/06/colorado-repeals-and-replaces-its-ai-act
  - https://www.consumerfinancemonitor.com/2026/05/12/colorado-rewrites-its-landmark-ai-law-unpacking-sb-26-189-and-what-it-means-for-businesses/
requirements:
  - {ref: "Consequential decisions", title: "Identify systems used in consequential decisions", controls: [C12]}
  - {ref: "Notice", title: "Consumers told an automated system is used", controls: [C9]}
  - {ref: "Adverse decisions", title: "Explanation and a route to correct data or appeal", controls: [C10, C8]}
  - {ref: "Records", title: "Keep records to show what the system did", controls: [C10]}
policy:
  transparency: {disclose: true}
"####,
    ),
    (
        "difc-reg10",
        r####"id: difc-reg10
name: DIFC Data Protection Law 2020 and Regulation 10 (autonomous and semi-autonomous systems)
jurisdiction: ae-difc
kind: regulation
binding: true
status: Regulation 10 fully enforced since 1 January 2026; certification requirements to follow
as_of: 2026-09-24
summary: >-
  The first AI-specific data-protection rule in the region. Anyone deploying or
  operating an autonomous or semi-autonomous system that processes personal data in
  the DIFC must give clear notice of it, keep its purposes human-defined, build it on
  stated principles, and — for high-risk processing — appoint an Autonomous Systems
  Officer.
sources:
  - https://www.difc.com/business/laws-and-regulations/legal-database/difc-laws/data-protection-law-difc-law-no-5-2020
  - https://www.mayerbrown.com/en/insights/publications/2026/01/ai-regulation-in-the-difc-personal-data-processed-through-autonomous-and-semi-autonomous-systems
requirements:
  - {ref: "Reg 10", title: "Clear notice that an autonomous system processes personal data", controls: [C9]}
  - {ref: "Reg 10", title: "Purposes defined or approved by humans", controls: [C7, C1]}
  - {ref: "Reg 10", title: "Designed on ethical, fair, transparent, secure and accountable principles", controls: [C3, C14]}
  - {ref: "Reg 10", title: "Human oversight of the system's outcomes", controls: [C8]}
  - {ref: "Reg 10", title: "Autonomous Systems Officer for high-risk processing", manual: true, controls: [C1]}
  - {ref: "Reg 10", title: "Certification of the system (requirements pending)", manual: true}
  - {ref: "DP Law — records", title: "Records of processing", controls: [C10, C13]}
  - {ref: "DP Law — DPIA", title: "DPIA for high-risk processing", controls: [C12]}
  - {ref: "DP Law Art 26-27", title: "Transfers to adequate jurisdictions or with safeguards", controls: [C6], note: "declare `residency.transfers['ae-difc']` from the Commissioner's adequacy list"}
  - {ref: "DP Law Art 41", title: "Breach notified to the Commissioner", controls: [C15]}
policy:
  transparency: {disclose: true, provenance: true, languages: [en, ar]}
  rules:
    - id: reg10-high-risk-oversight
      on: tool
      match: {risk: high, jurisdiction: [ae-difc], tags: [consequential, irreversible]}
      effect: require_approval
      reason: "DIFC Regulation 10: a person oversees high-risk autonomous processing"
      controls: [C8]
incidents:
  personal_data_breach:
    - {notify: DIFC Commissioner of Data Protection, within: 0, basis: "Art 41 — as soon as practicable", min_severity: medium}
"####,
    ),
    (
        "dora",
        r####"id: dora
name: Digital Operational Resilience Act (EU) 2022/2554
jurisdiction: eu
kind: regulation
binding: true
status: applies since 17 January 2025 (financial entities)
as_of: 2026-09-24
summary: >-
  ICT risk management for EU financial entities. Model providers are ICT third-party
  service providers: they belong in the register of information, and major ICT
  incidents are reported on a fixed clock.
sources:
  - https://eur-lex.europa.eu/eli/reg/2022/2554/oj
requirements:
  - {ref: "Art 6", title: ICT risk management framework, controls: [C3, C14]}
  - {ref: "Art 8", title: Identification of ICT assets and dependencies, controls: [C13]}
  - {ref: "Art 12", title: Logging and backup, controls: [C10]}
  - {ref: "Art 17", title: ICT incident management process, controls: [C15]}
  - {ref: "Art 19", title: "Major ICT incidents reported (initial, intermediate, final)", controls: [C15]}
  - {ref: "Art 28(3)", title: Register of information on ICT third-party arrangements, controls: [C13], note: "gov.inventory.third_parties()"}
  - {ref: "Art 28(4)", title: Due diligence before contracting ICT third parties, manual: true}
  - {ref: "Art 30", title: Key contractual provisions with ICT providers, manual: true}
policy:
  tool_drift: deny
incidents:
  ict_incident:
    - {notify: competent authority (initial), within: 4, basis: "RTS: 4 h after classification as major, at most 24 h after detection", min_severity: high}
    - {notify: competent authority (intermediate), within: 72, basis: "RTS: 72 h after the initial notification", min_severity: high}
    - {notify: competent authority (final), within: 720, basis: "RTS: one month", min_severity: high}
"####,
    ),
    (
        "eu-ai-act",
        r####"id: eu-ai-act
name: EU Artificial Intelligence Act (Regulation (EU) 2024/1689)
jurisdiction: eu
kind: regulation
binding: true
status: >-
  Art 5 prohibitions and GPAI rules in force (Feb/Aug 2025); Art 50 transparency from
  2 Aug 2026 (machine-readable marking for systems already on the market: 2 Dec 2026);
  high-risk obligations deferred by the Digital Omnibus (in force 27 Jul 2026) to
  2 Dec 2027 (Annex III) and 2 Aug 2028 (Annex I)
as_of: 2026-09-24
summary: >-
  Risk-based rules for AI systems. For agents the practical duties are: refuse prohibited
  practices, tell people they are dealing with AI, mark generated content, and — for
  high-risk uses — log automatically, keep humans able to oversee and stop the system,
  and report serious incidents.
sources:
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj
  - https://artificialintelligenceact.eu/transparency-rules-article-50/
  - https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/
requirements:
  - {ref: "Art 5", title: Prohibited AI practices are not deployed, controls: [C12]}
  - {ref: "Art 4", title: AI literacy of staff operating the system, manual: true}
  - {ref: "Art 6 + Annex III", title: High-risk classification of each use, controls: [C12, C1]}
  - {ref: "Art 9", title: Risk management system across the lifecycle, controls: [C12, C14, C3]}
  - {ref: "Art 10", title: Data governance for the data the system uses, controls: [C4, C5, C7]}
  - {ref: "Art 11 + Annex IV", title: Technical documentation, controls: [C13, C16]}
  - {ref: "Art 12", title: Automatic recording of events (logging), controls: [C10]}
  - {ref: "Art 13", title: Instructions for use and transparency to deployers, controls: [C9, C13]}
  - {ref: "Art 14", title: "Human oversight — understand, override and stop", controls: [C8]}
  - {ref: "Art 15", title: "Accuracy, robustness and cybersecurity", controls: [C14, C13]}
  - {ref: "Art 26(1)-(5)", title: "Deployer: use per instructions, assign competent oversight", controls: [C1, C8]}
  - {ref: "Art 26(6)", title: "Deployer: keep automatically generated logs at least six months", controls: [C10]}
  - {ref: "Art 26(7)", title: "Deployer: inform workers before workplace use", manual: true}
  - {ref: "Art 27", title: "Fundamental rights impact assessment (public bodies, credit, insurance)", controls: [C12], note: "gov.impact_assessment(agent, 'fria') drafts it"}
  - {ref: "Art 49", title: Registration of high-risk systems in the EU database, manual: true}
  - {ref: "Art 50(1)", title: People are told they are interacting with an AI system, controls: [C9]}
  - {ref: "Art 50(2)", title: Generated content is marked in a machine-readable way, controls: [C9]}
  - {ref: "Art 50(4)", title: Deep fakes and public-interest text are disclosed, controls: [C9]}
  - {ref: "Art 72", title: Post-market monitoring, controls: [C14, C16]}
  - {ref: "Art 73", title: Serious incidents reported to market surveillance authorities, controls: [C15]}
policy:
  transparency: {disclose: true, provenance: true, languages: [en]}
  log_retention_days: 183
  rules:
    - id: art14-oversight-high-risk
      on: tool
      match: {risk: high, tags: [consequential, irreversible]}
      effect: require_approval
      reason: "EU AI Act Art 14: a person reviews consequential actions of a high-risk system"
      controls: [C8]
incidents:
  serious_incident:
    - {notify: market surveillance authority, within: 360, basis: "Art 73(2) — 15 days", min_severity: medium}
    - {notify: market surveillance authority, within: 240, basis: "Art 73(4) — 10 days where a death is involved", min_severity: high}
    - {notify: market surveillance authority, within: 48, basis: "Art 73(3) — 2 days: widespread infringement or critical infrastructure", min_severity: critical}
"####,
    ),
    (
        "gdpr",
        r####"id: gdpr
name: General Data Protection Regulation (EU) 2016/679
jurisdiction: eu
kind: regulation
binding: true
status: in force since 25 May 2018
as_of: 2026-09-24
summary: >-
  Personal data of people in the EU/EEA: lawful basis and purpose limitation,
  minimisation, transfers only to adequate or safeguarded destinations, rights of
  access and erasure, safeguards on solely automated decisions, 72-hour breach notice.
sources:
  - https://eur-lex.europa.eu/eli/reg/2016/679/oj
  - https://commission.europa.eu/law/law-topic/data-protection/international-dimension-data-protection/adequacy-decisions_en
requirements:
  - {ref: "Art 5(1)(b)", title: Purpose limitation, controls: [C7]}
  - {ref: "Art 5(1)(c)", title: Data minimisation, controls: [C5, C4]}
  - {ref: "Art 5(1)(e)", title: Storage limitation, controls: [C11], note: "set data_retention_days and run gov.sweep()"}
  - {ref: "Art 5(2)", title: Accountability — be able to demonstrate compliance, controls: [C10, C16]}
  - {ref: "Art 6", title: Lawful basis for each processing purpose, manual: true}
  - {ref: "Art 9", title: Special-category data only with an Art 9(2) condition, controls: [C4, C7]}
  - {ref: "Art 13-14", title: Information given to data subjects, controls: [C9]}
  - {ref: "Art 15", title: Right of access, controls: [C11]}
  - {ref: "Art 17", title: Right to erasure, controls: [C11]}
  - {ref: "Art 22", title: Safeguards for solely automated decisions with legal effect, controls: [C8]}
  - {ref: "Art 25", title: Data protection by design and by default, controls: [C5, C3]}
  - {ref: "Art 28", title: Processor contracts with model providers, manual: true}
  - {ref: "Art 30", title: Records of processing activities, controls: [C13, C10]}
  - {ref: "Art 32", title: Security of processing, controls: [C5, C10, C14]}
  - {ref: "Art 33", title: Breach notified to the supervisory authority within 72 hours, controls: [C15]}
  - {ref: "Art 35", title: Data protection impact assessment, controls: [C12], note: "gov.impact_assessment(agent) drafts it"}
  - {ref: "Art 37", title: Data protection officer where required, manual: true}
  - {ref: "Art 44-49", title: Transfers only to adequate countries or with safeguards, controls: [C6]}
policy:
  residency:
    # Adequacy decisions. The EEA is "eu". The US is adequate only for organisations
    # certified under the EU-US Data Privacy Framework — add "us" yourself if your
    # model provider is certified (or you rely on SCCs plus a transfer assessment).
    transfers:
      eu: [eu, uk, ch, jp, kr, ca, il, nz, ar, uy, ad, fo, gg, im, je]
    applies_to: [personal]
  rules:
    - id: art22-human-review
      on: tool
      match: {tags: [consequential]}
      effect: require_approval
      reason: "GDPR Art 22: a decision with legal or similarly significant effect gets human involvement"
      controls: [C8]
    - id: art9-special-category-memory
      on: memory
      match: {data: [special_category]}
      when: "'special_category' not in principal.consents"
      effect: deny
      reason: "GDPR Art 9: special-category data is not kept without explicit consent"
      controls: [C7]
incidents:
  personal_data_breach:
    - {notify: supervisory authority, within: 72, basis: "Art 33 — 72 hours", min_severity: medium}
    - {notify: affected data subjects, within: 72, basis: "Art 34 — without undue delay when high risk", min_severity: high}
"####,
    ),
    (
        "india-dpdp",
        r####"id: india-dpdp
name: India Digital Personal Data Protection Act 2023 and DPDP Rules 2025
jurisdiction: in
kind: law
binding: true
status: >-
  Rules notified 13 Nov 2025; consent-manager framework operative 13 Nov 2026; main
  processing, rights and penalty provisions from 13 May 2027
as_of: 2026-09-24
summary: >-
  Notice and consent, purpose-bound processing, verifiable parental consent for
  children, erasure when the purpose ends, breach intimation to the Board and every
  affected person. Transfers are allowed except to countries the government restricts.
sources:
  - https://www.meity.gov.in/data-protection-framework
  - https://www.india-briefing.com/news/india-dpdp-compliance-timeline-enforcement-2026-27-44740.html/
requirements:
  - {ref: "s 5", title: "Notice to the data principal", controls: [C9]}
  - {ref: "s 6", title: "Consent — free, specific, informed; withdrawable", manual: true}
  - {ref: "s 6(7)-(9)", title: "Consent managers (registered from 13 Nov 2026)", manual: true}
  - {ref: "s 8(5)", title: "Reasonable security safeguards", controls: [C5, C10]}
  - {ref: "s 8(6) + Rule 7", title: "Breach intimated to the Board and affected principals", controls: [C15]}
  - {ref: "s 8(7)", title: "Erasure once the purpose is served", controls: [C11]}
  - {ref: "s 9", title: "Verifiable parental consent; no tracking or targeting of children", controls: [C4, C7]}
  - {ref: "s 10", title: "Significant data fiduciaries — DPIA, audit, DPO", controls: [C12], manual: true}
  - {ref: "s 11-12", title: "Rights to access, correction and erasure", controls: [C11]}
  - {ref: "s 16", title: "Transfers except to restricted countries", controls: [C6]}
policy:
  residency:
    transfers:
      in: ["*"]
  rules:
    - id: s9-children
      on: [memory, egress]
      match: {data: [children], jurisdiction: [in]}
      when: "'parental' not in principal.consents"
      effect: deny
      reason: "DPDP s 9: a child's data needs verifiable parental consent"
      controls: [C7]
incidents:
  personal_data_breach:
    - {notify: Data Protection Board (intimation), within: 0, basis: "Rule 7 — without delay", min_severity: medium}
    - {notify: Data Protection Board (detailed report), within: 72, basis: "Rule 7 — within 72 hours", min_severity: medium}
    - {notify: affected data principals, within: 0, basis: "s 8(6) — without delay", min_severity: medium}
"####,
    ),
    (
        "iso-42001",
        r####"id: iso-42001
name: ISO/IEC 42001:2023 — AI management system
jurisdiction: global
kind: standard
binding: false
status: certifiable international standard
as_of: 2026-09-24
summary: >-
  The management-system standard for AI. Annex A controls are where a runtime can
  supply evidence; the clauses on leadership, planning and review are organisational.
sources:
  - https://www.iso.org/standard/81230.html
requirements:
  - {ref: "Clause 5", title: "Leadership, AI policy and roles", manual: true}
  - {ref: "Clause 6.1", title: "AI risk assessment and treatment", controls: [C12]}
  - {ref: "Clause 9", title: "Monitoring, internal audit and management review", manual: true, controls: [C16]}
  - {ref: "A.2", title: "Policies related to AI", controls: [C3]}
  - {ref: "A.3", title: "Internal organisation — roles and responsibilities", controls: [C1]}
  - {ref: "A.4", title: "Resources for AI systems documented", controls: [C13]}
  - {ref: "A.5", title: "Assessing impacts of AI systems", controls: [C12]}
  - {ref: "A.6", title: "AI system life cycle — including event logs", controls: [C10, C14]}
  - {ref: "A.7", title: "Data for AI systems — quality, provenance, preparation", controls: [C4, C5]}
  - {ref: "A.8", title: "Information for interested parties", controls: [C9]}
  - {ref: "A.9", title: "Use of AI systems — intended use, human oversight", controls: [C7, C8]}
  - {ref: "A.10", title: "Third-party and customer relationships", controls: [C13]}
"####,
    ),
    (
        "japan-appi",
        r####"id: japan-appi
name: Japan Act on the Protection of Personal Information and AI Promotion Act
jurisdiction: jp
kind: law
binding: true
status: APPI (amended 2022) in force; AI Promotion Act enacted May 2025 (soft law, with AI Basic Plan)
as_of: 2026-09-24
summary: >-
  Overseas transfers need consent unless the destination has an equivalent system (the
  EU and UK are recognised) or the recipient keeps equivalent standards. Serious
  breaches get a prompt preliminary report and a final report to the PPC.
sources:
  - https://www.ppc.go.jp/en/legal/
requirements:
  - {ref: "APPI Art 17-18", title: "Use limited to the specified purpose", controls: [C7]}
  - {ref: "APPI Art 20(2)", title: "Special care-required personal information needs consent", controls: [C4, C7]}
  - {ref: "APPI Art 23", title: "Security control measures", controls: [C5, C10]}
  - {ref: "APPI Art 26", title: "Serious breaches reported to the PPC and individuals", controls: [C15]}
  - {ref: "APPI Art 28", title: "Overseas transfers — consent, equivalent country or equivalent standards", controls: [C6]}
  - {ref: "APPI Art 33-35", title: "Disclosure, correction and cessation of use", controls: [C11]}
  - {ref: "AI Promotion Act", title: "Cooperate with the government's AI measures and guidelines", manual: true}
policy:
  residency:
    transfers:
      jp: [jp, eu, uk]
    applies_to: [personal]
incidents:
  personal_data_breach:
    - {notify: Personal Information Protection Commission (preliminary), within: 120, basis: "Art 26 — promptly (about 3-5 days)", min_severity: medium}
    - {notify: Personal Information Protection Commission (final), within: 720, basis: "Art 26 — within 30 days (60 if malicious)", min_severity: medium}
"####,
    ),
    (
        "korea-ai-basic",
        r####"id: korea-ai-basic
name: South Korea Framework Act on AI (AI Basic Act) and PIPA
jurisdiction: kr
kind: law
binding: true
status: >-
  AI Basic Act and Enforcement Decree in force 22 January 2026; MSIT grace period of at
  least one year before fines, except serious harm
as_of: 2026-09-24
summary: >-
  Tell users in advance that AI is used and label generated output; high-impact AI
  (hiring, lending, healthcare, energy, public services, education) needs risk
  management, explanation, human oversight and records. Foreign operators above the
  thresholds appoint a domestic representative. Personal data breaches go to the PIPC
  within 72 hours under PIPA.
sources:
  - https://www.cooley.com/news/insight/2026/2026-01-27-south-koreas-ai-basic-act-overview-and-key-takeaways
  - https://www.loc.gov/item/global-legal-monitor/2026-02-20/south-korea-comprehensive-ai-legal-framework-takes-effect
requirements:
  - {ref: "Art 31(1)", title: "Users told in advance that the product uses AI", controls: [C9]}
  - {ref: "Art 31(2)-(3)", title: "Generated output labelled as AI-generated", controls: [C9]}
  - {ref: "Art 33", title: "Whether the system is high-impact is determined", controls: [C12]}
  - {ref: "Art 34", title: "High-impact: risk management plan", controls: [C12, C14]}
  - {ref: "Art 34", title: "High-impact: explanation of outcomes", controls: [C10, C9]}
  - {ref: "Art 34", title: "High-impact: human oversight and supervision", controls: [C8]}
  - {ref: "Art 34", title: "High-impact: documents kept on safety and reliability", controls: [C10, C13]}
  - {ref: "Art 35", title: "Impact assessment on fundamental rights", controls: [C12]}
  - {ref: "Art 36", title: "Domestic representative for foreign operators", manual: true}
  - {ref: "PIPA Art 28-8", title: "Overseas transfer only on a lawful basis", controls: [C6]}
  - {ref: "PIPA Art 34", title: "Breach notified to data subjects and the PIPC", controls: [C15]}
policy:
  transparency: {disclose: true, label_output: true, label_for: [kr], provenance: true, languages: [ko]}
  rules:
    - id: art34-high-impact-oversight
      on: tool
      match: {risk: high, tags: [consequential]}
      effect: require_approval
      reason: "Korea AI Basic Act Art 34: human oversight of high-impact AI"
      controls: [C8]
incidents:
  personal_data_breach:
    - {notify: Personal Information Protection Commission, within: 72, basis: "PIPA Art 34 — 72 hours", min_severity: medium}
"####,
    ),
    (
        "ksa-pdpl",
        r####"id: ksa-pdpl
name: Saudi Personal Data Protection Law and Transfer Regulation
jurisdiction: sa
kind: law
binding: true
status: enforced since 14 September 2024; Transfer Regulation amended September 2024
as_of: 2026-09-24
summary: >-
  The strictest residency posture in the GCC. Personal data stays in the Kingdom unless
  the destination is adequate or safeguards apply (standard contractual clauses, binding
  common rules, certification), with a documented risk assessment and a record of each
  transfer. Breaches go to SDAIA within 72 hours.
sources:
  - https://sdaia.gov.sa/en/SDAIA/about/Documents/Personal%20Data%20English%20V2-23April2023-%20Reviewed-.pdf
  - https://www.kslaw.com/news-and-insights/international-personal-data-transfers-under-saudi-arabias-data-protection-law
requirements:
  - {ref: "Art 10-11", title: "Collection limited to what the purpose needs", controls: [C5, C7]}
  - {ref: "Art 4", title: "Rights of access, correction and destruction", controls: [C11]}
  - {ref: "Art 18", title: "Destruction once the purpose is fulfilled", controls: [C11]}
  - {ref: "Art 19", title: "Organisational and technical safeguards", controls: [C5, C10, C14]}
  - {ref: "Art 20", title: "Breaches notified to SDAIA", controls: [C15]}
  - {ref: "Art 22", title: "Impact assessment of processing", controls: [C12]}
  - {ref: "Art 29 + Transfer Regulation", title: "Transfers outside the Kingdom only on an approved basis", controls: [C6]}
  - {ref: "Transfer Regulation", title: "Risk assessment and records for each transfer", controls: [C6, C10]}
  - {ref: "Implementing Regulation — DPO", title: "Personal data protection officer where required", manual: true}
  - {ref: "Implementing Regulation — records", title: "Record of processing activities", controls: [C13, C10]}
policy:
  residency:
    transfers:
      sa: [sa]
    applies_to: [personal]
incidents:
  personal_data_breach:
    - {notify: SDAIA (National Data Governance Platform), within: 72, basis: "Implementing Regulation — 72 hours", min_severity: medium}
    - {notify: affected data subjects, within: 72, basis: "Implementing Regulation — without undue delay where harm is likely", min_severity: high}
"####,
    ),
    (
        "nis2",
        r####"id: nis2
name: NIS2 Directive (EU) 2022/2555
jurisdiction: eu
kind: law
binding: true
status: transposition deadline 17 Oct 2024; national laws in force or pending by member state
as_of: 2026-09-24
summary: Cyber-security risk management and incident reporting for essential and important entities.
sources:
  - https://eur-lex.europa.eu/eli/dir/2022/2555/oj
requirements:
  - {ref: "Art 21", title: Cyber-security risk-management measures, controls: [C3, C13, C14]}
  - {ref: "Art 21(2)(d)", title: Supply-chain security, controls: [C13]}
  - {ref: "Art 23", title: Significant incidents reported on the NIS2 clock, controls: [C15]}
  - {ref: "Art 20", title: Management body approves and oversees measures, manual: true}
incidents:
  security_incident:
    - {notify: CSIRT or competent authority (early warning), within: 24, basis: "Art 23(4)(a) — 24 hours", min_severity: high}
    - {notify: CSIRT or competent authority (notification), within: 72, basis: "Art 23(4)(b) — 72 hours", min_severity: high}
    - {notify: CSIRT or competent authority (final report), within: 720, basis: "Art 23(4)(d) — one month", min_severity: high}
"####,
    ),
    (
        "oman-pdpl",
        r####"id: oman-pdpl
name: Oman Personal Data Protection Law (Royal Decree 6/2022) and Executive Regulations
jurisdiction: om
kind: law
binding: true
status: in force since 13 February 2023; Executive Regulations 2024
as_of: 2026-09-24
summary: Consent-based regime with Ministry permits for sensitive data and breach reporting.
sources:
  - https://www.mtcit.gov.om/
requirements:
  - {ref: "Law — sensitive data", title: "Sensitive data (health, genetic, biometric, religion...) needs a Ministry permit", manual: true, controls: [C4]}
  - {ref: "Law — security", title: "Security measures", controls: [C5, C10]}
  - {ref: "Law — breach", title: "Breach notified to the Ministry and the data subject", controls: [C15]}
  - {ref: "Law — transfers", title: "Transfers subject to the Executive Regulations' conditions", controls: [C6]}
  - {ref: "Law — rights", title: "Rights including erasure", controls: [C11]}
incidents:
  personal_data_breach:
    - {notify: Ministry of Transport, Communications and IT, within: 72, basis: "Executive Regulations — 72 hours", min_severity: medium}
"####,
    ),
    (
        "owasp-agentic",
        r####"id: owasp-agentic
name: OWASP Top 10 for Agentic Applications 2026
jurisdiction: global
kind: framework
binding: false
status: published 9 December 2025 by the OWASP GenAI Security Project
as_of: 2026-09-24
summary: >-
  The ten highest-impact risks specific to agents that plan, hold memory, call tools
  and act with delegated authority. This pack adds the runtime defences directly.
sources:
  - https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/
requirements:
  - {ref: "ASI01", title: "Agent goal hijack", controls: [C14, C8]}
  - {ref: "ASI02", title: "Tool misuse and exploitation", controls: [C2, C3, C8]}
  - {ref: "ASI03", title: "Agent identity and privilege abuse", controls: [C1, C2]}
  - {ref: "ASI04", title: "Agentic supply-chain compromise", controls: [C13]}
  - {ref: "ASI05", title: "Unexpected code execution", controls: [C2, C8], note: "workspace shell is off unless allow_shell=True and asks every time"}
  - {ref: "ASI06", title: "Memory and context poisoning", controls: [C14, C5]}
  - {ref: "ASI07", title: "Insecure inter-agent communication", controls: [C2, C10]}
  - {ref: "ASI08", title: "Cascading agent failures", controls: [C14]}
  - {ref: "ASI09", title: "Human-agent trust exploitation", controls: [C9, C8]}
  - {ref: "ASI10", title: "Rogue agents", controls: [C14, C8, C13]}
policy:
  max_delegation_depth: 3
  rules:
    - id: asi01-untrusted-context
      on: tool
      match: {tags: [irreversible, payments, external_comms, data_export, consequential]}
      when: "run.untrusted"
      effect: require_approval
      reason: "OWASP ASI01/ASI06: the run has read content with injection signals; a person confirms before a sensitive action"
      controls: [C8, C14]
    - id: asi06-no-injected-memory
      on: memory
      when: "run.untrusted"
      effect: deny
      reason: "OWASP ASI06: nothing is written to long-term memory from a run that read suspected injection"
      controls: [C14]
"####,
    ),
    (
        "qatar-pdppl",
        r####"id: qatar-pdppl
name: Qatar Personal Data Privacy Protection Law (Law No. 13 of 2016)
jurisdiction: qa
kind: law
binding: true
status: in force; NCSA/NCGAA guidelines
as_of: 2026-09-24
summary: >-
  Consent-based regime with a permit requirement for data of a special nature.
  Article 15 forbids restricting cross-border flows unless processing breaches the law
  or could harm the individual — so transfers are not residency-restricted by default.
sources:
  - https://www.almeezan.qa/LawPage.aspx?id=7121&language=en
requirements:
  - {ref: "Art 4", title: "Processing is honest and lawful and within its purpose", controls: [C7]}
  - {ref: "Art 5-6", title: "Rights including withdrawal and erasure", controls: [C11]}
  - {ref: "Art 11", title: "Precautions to protect the data", controls: [C5, C10]}
  - {ref: "Art 14", title: "Breach notified to the competent department and the individual", controls: [C15]}
  - {ref: "Art 16", title: "Data of a special nature only with a permit", manual: true, controls: [C4]}
  - {ref: "Art 15", title: "Cross-border flows are not restricted without cause", controls: [C6]}
policy:
  residency:
    transfers:
      qa: ["*"]
  rules:
    - id: special-nature-needs-permit
      on: egress
      match: {data: [special_category], jurisdiction: [qa]}
      when: "'qa_special_permit' not in principal.consents"
      effect: redact
      redact: [special_category]
      reason: "Qatar PDPPL Art 16: data of a special nature is processed only under a permit"
      controls: [C5]
incidents:
  personal_data_breach:
    - {notify: National Cyber Governance and Assurance Affairs, within: 72, basis: "Art 14 and NCGAA guideline — 72 hours", min_severity: medium}
"####,
    ),
    (
        "sdaia",
        r####"id: sdaia
name: SDAIA AI Ethics Principles, Generative AI Guidelines and AI Adoption Framework
jurisdiction: sa
kind: guidance
binding: false
status: >-
  voluntary; Ethics Principles 2023, GenAI Guidelines 2024, AI Adoption Framework
  Nov 2025 — used in government procurement and maturity assessments
as_of: 2026-09-24
summary: >-
  Saudi Arabia's AI governance guidance. Not binding, but procurement and the AI
  adoption maturity levels expect evidence against each principle.
sources:
  - https://sdaia.gov.sa/en/SDAIA/about/Documents/ai-principles.pdf
  - https://vision2030.ai/regulation/saudi-arabia-ai-regulation-sdaia/
requirements:
  - {ref: "Principle 1", title: "Fairness", manual: true, note: "evaluate outputs across groups with agent_harness.Evaluator"}
  - {ref: "Principle 2", title: "Privacy and security", controls: [C4, C5, C6]}
  - {ref: "Principle 3", title: "Humanity — humans stay in control", controls: [C8]}
  - {ref: "Principle 4", title: "Social and environmental benefits", manual: true}
  - {ref: "Principle 5", title: "Reliability and safety", controls: [C14]}
  - {ref: "Principle 6", title: "Transparency and explainability", controls: [C9, C10]}
  - {ref: "Principle 7", title: "Accountability and responsibility", controls: [C1, C10]}
  - {ref: "GenAI Guidelines", title: "Disclose generated content and keep humans reviewing it", controls: [C9, C8]}
  - {ref: "Adoption Framework", title: "Inventory and maturity evidence", controls: [C13, C16]}
policy:
  transparency: {disclose: true, provenance: true, languages: [ar, en]}
"####,
    ),
    (
        "singapore-agentic",
        r####"id: singapore-agentic
name: Singapore Model AI Governance Framework for Agentic AI (IMDA)
jurisdiction: sg
kind: framework
binding: false
status: launched 22 January 2026; updated 20 May 2026
as_of: 2026-09-24
summary: >-
  The first governance framework written for AI agents. Four dimensions: assess and
  bound the risks up front; make humans meaningfully accountable; implement technical
  controls and processes across the lifecycle; enable end-user responsibility.
sources:
  - https://www.imda.gov.sg/resources/press-releases-factsheets-and-speeches/press-releases/2026/new-model-ai-governance-framework-for-agentic-ai
  - https://www.bakermckenzie.com/en/insight/publications/2026/06/singapore-imda-updates-model-ai-governance-framework-for-agentic-ai
requirements:
  - {ref: "Dimension 1", title: "Assess risk per use case before deployment", controls: [C12]}
  - {ref: "Dimension 1", title: "Bound the agent — limit its tools, data and autonomy", controls: [C2, C7]}
  - {ref: "Dimension 1", title: "Agent identity and permissions traceable to a human", controls: [C1, C2]}
  - {ref: "Dimension 2", title: "Named humans accountable across the lifecycle", controls: [C1]}
  - {ref: "Dimension 2", title: "Human checkpoints for significant or irreversible actions", controls: [C8]}
  - {ref: "Dimension 3", title: "Testing before deployment", manual: true, note: "agent_harness.Evaluator golden tasks"}
  - {ref: "Dimension 3", title: "Continuous monitoring and logging of agent behaviour", controls: [C14, C10]}
  - {ref: "Dimension 3", title: "Controls on tools, inter-agent communication and supply chain", controls: [C2, C13]}
  - {ref: "Dimension 3", title: "Ability to interrupt or shut down an agent", controls: [C8]}
  - {ref: "Dimension 4", title: "Users know they are dealing with an agent and what it can do", controls: [C9]}
policy:
  transparency: {disclose: true}
  max_delegation_depth: 3
  rules:
    - id: irreversible-actions-need-a-human
      on: tool
      match: {tags: [irreversible]}
      effect: require_approval
      reason: "Singapore agentic framework: human checkpoint before irreversible actions"
      controls: [C8]
"####,
    ),
    (
        "singapore-pdpa",
        r####"id: singapore-pdpa
name: Singapore Personal Data Protection Act 2012
jurisdiction: sg
kind: law
binding: true
status: in force; mandatory breach notification since 2021
as_of: 2026-09-24
summary: >-
  Consent, purpose and notification obligations; the Transfer Limitation Obligation
  requires a comparable standard of protection abroad (usually by contract); notifiable
  breaches go to the PDPC within 3 calendar days of assessment.
sources:
  - https://sso.agc.gov.sg/Act/PDPA2012
requirements:
  - {ref: "s 13-17", title: "Consent obligation", manual: true}
  - {ref: "s 18", title: "Purpose limitation", controls: [C7]}
  - {ref: "s 20", title: "Notification of purposes", controls: [C9]}
  - {ref: "s 21-22", title: "Access and correction", controls: [C11]}
  - {ref: "s 24", title: "Protection obligation", controls: [C5, C10]}
  - {ref: "s 25", title: "Retention limitation", controls: [C11], note: "set data_retention_days and run gov.sweep()"}
  - {ref: "s 26", title: "Transfer limitation — comparable protection abroad", controls: [C6], note: "by contract; declare destinations in residency.transfers['sg'] to enforce"}
  - {ref: "s 26A-26E", title: "Notifiable breaches reported to the PDPC", controls: [C15]}
  - {ref: "s 11(3)", title: "Data protection officer", manual: true}
incidents:
  personal_data_breach:
    - {notify: Personal Data Protection Commission, within: 72, basis: "s 26D — 3 calendar days after assessing it notifiable", min_severity: medium}
"####,
    ),
    (
        "texas-traiga",
        r####"id: texas-traiga
name: Texas Responsible Artificial Intelligence Governance Act (HB 149)
jurisdiction: us
kind: law
binding: true
status: in force since 1 January 2026; enforced by the Texas Attorney General
as_of: 2026-09-24
summary: >-
  Prohibits developing or deploying AI with intent to incite self-harm, harm or crime,
  to discriminate unlawfully, or to produce certain deepfakes and CSAM; government
  agencies and health-care providers disclose AI use to consumers.
sources:
  - https://www.bakerbotts.com/thought-leadership/publications/2025/july/texas-enacts-responsible-ai-governance-act-what-companies-need-to-know
requirements:
  - {ref: "Prohibited practices", title: "No manipulation to self-harm, harm or crime", controls: [C12]}
  - {ref: "Prohibited practices", title: "No intent to discriminate against a protected class", manual: true}
  - {ref: "Prohibited practices", title: "No unlawful deepfakes or CSAM", controls: [C12]}
  - {ref: "Disclosure", title: "Government and health-care deployers disclose AI interaction", controls: [C9]}
  - {ref: "AG inquiries", title: "Be able to produce documentation on request", controls: [C13, C16]}
"####,
    ),
    (
        "uae-pdpl",
        r####"id: uae-pdpl
name: UAE Personal Data Protection Law (Federal Decree-Law No. 45 of 2021)
jurisdiction: ae
kind: law
binding: true
status: in force; check the status of its executive regulations and Data Office adequacy list
as_of: 2026-09-24
summary: >-
  The federal data-protection law for the UAE mainland (DIFC and ADGM have their own).
  Consent or another basis, transfers only to adequate or safeguarded destinations,
  immediate breach notification. Health data is also subject to Federal Law No. 2 of
  2019, which keeps it in the UAE unless an exception applies.
sources:
  - https://uaelegislation.gov.ae/en/legislations/1972
  - https://millennial.ae/uae-gcc-ai-regulation-2026/
requirements:
  - {ref: "Art 4-5", title: "Lawful processing, consent and its conditions", manual: true}
  - {ref: "Art 5", title: "Processing limited to the specified purpose", controls: [C7]}
  - {ref: "Art 7", title: "Controller keeps a record of processing", controls: [C13, C10]}
  - {ref: "Art 9", title: "Breaches notified to the Data Office", controls: [C15]}
  - {ref: "Art 13-15", title: "Rights of access, correction and erasure", controls: [C11]}
  - {ref: "Art 18", title: "Objection to automated processing", controls: [C8]}
  - {ref: "Art 20", title: "Security and confidentiality of data", controls: [C5, C10]}
  - {ref: "Art 21", title: "Data protection impact assessment for high-risk processing", controls: [C12]}
  - {ref: "Art 22-23", title: "Cross-border transfers only to adequate or safeguarded destinations", controls: [C6]}
  - {ref: "Federal Law 2/2019 Art 13", title: "Health data stored and processed inside the UAE", controls: [C6]}
policy:
  residency:
    # Until the Data Office publishes its adequacy list, keep UAE personal data in
    # the UAE; add destinations once you have a contractual or adequacy basis.
    transfers:
      ae: [ae]
    applies_to: [personal]
  rules:
    - id: health-data-stays-in-uae
      on: egress
      match: {data: [health], jurisdiction: [ae]}
      when: "provider.region not in ['ae', 'on_prem']"
      effect: reroute
      reason: "UAE Federal Law 2/2019: health data is processed inside the UAE"
      controls: [C6]
incidents:
  personal_data_breach:
    - {notify: UAE Data Office, within: 0, basis: "Art 9 — immediately upon becoming aware", min_severity: medium}
"####,
    ),
    (
        "uk-gdpr",
        r####"id: uk-gdpr
name: UK GDPR, Data Protection Act 2018 and Data (Use and Access) Act 2025
jurisdiction: uk
kind: law
binding: true
status: in force; DUAA 2025 provisions commencing in stages
as_of: 2026-09-24
summary: >-
  GDPR as retained in UK law, amended by the Data (Use and Access) Act 2025, which
  relaxes automated decision-making rules while keeping safeguards (notice, human
  review on request, the right to contest).
sources:
  - https://www.legislation.gov.uk/ukpga/2025/18/contents
  - https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/
requirements:
  - {ref: "Art 5", title: Data protection principles, controls: [C5, C7, C10]}
  - {ref: "Art 15 / 17", title: Rights of access and erasure, controls: [C11]}
  - {ref: "Art 22A-22D", title: Safeguards on significant automated decisions, controls: [C8, C9]}
  - {ref: "Art 33", title: Breach reported to the ICO within 72 hours, controls: [C15]}
  - {ref: "Art 35", title: DPIA for high-risk processing, controls: [C12]}
  - {ref: "Art 44-49", title: Restricted transfers, controls: [C6]}
  - {ref: "Art 37", title: DPO where required, manual: true}
policy:
  residency:
    # UK adequacy regulations. The US "data bridge" covers only DPF-certified
    # organisations — add "us" yourself when it applies.
    transfers:
      uk: [uk, eu, ch, jp, kr, ca, il, nz, ar, uy, ad, fo, gg, im, je]
    applies_to: [personal]
  rules:
    - id: art22-significant-decisions
      on: tool
      match: {tags: [consequential]}
      when: "'special_category' in data.classes"
      effect: require_approval
      reason: "UK GDPR Art 22B: significant decisions on special-category data need human involvement"
      controls: [C8]
incidents:
  personal_data_breach:
    - {notify: Information Commissioner's Office, within: 72, basis: "Art 33 — 72 hours", min_severity: medium}
"####,
    ),
    (
        "us-nist-rmf",
        r####"id: us-nist-rmf
name: NIST AI Risk Management Framework 1.0 and Generative AI Profile (NIST AI 600-1)
jurisdiction: us
kind: framework
binding: false
status: voluntary; widely required in US federal and enterprise procurement
as_of: 2026-09-24
summary: Four functions — Govern, Map, Measure, Manage — with the GenAI profile's risk list.
sources:
  - https://www.nist.gov/itl/ai-risk-management-framework
  - https://doi.org/10.6028/NIST.AI.600-1
requirements:
  - {ref: "GOVERN 1", title: "Policies, processes and practices are in place and transparent", controls: [C3, C16]}
  - {ref: "GOVERN 2", title: "Accountability structures — roles and responsibilities", controls: [C1]}
  - {ref: "GOVERN 6", title: "Third-party and supply-chain risks are addressed", controls: [C13]}
  - {ref: "MAP 1", title: "Context and intended purpose are established", controls: [C1, C7]}
  - {ref: "MAP 4", title: "Risks and benefits mapped for all components, including third parties", controls: [C13, C12]}
  - {ref: "MAP 5", title: "Impacts on individuals and groups characterised", controls: [C12]}
  - {ref: "MEASURE 2", title: "System evaluated for trustworthy characteristics", manual: true, note: "agent_harness.Evaluator"}
  - {ref: "MEASURE 3", title: "Risks tracked over time", controls: [C14, C10]}
  - {ref: "MANAGE 2", title: "Mechanisms to supersede, disengage or deactivate the system", controls: [C8]}
  - {ref: "MANAGE 4", title: "Incidents and errors communicated and responded to", controls: [C15]}
  - {ref: "AI 600-1", title: "Data privacy, information security and value-chain risks of GenAI", controls: [C4, C5, C13]}
"####,
    ),
    (
        "vietnam-ai",
        r####"id: vietnam-ai
name: Vietnam Law on Artificial Intelligence and Personal Data Protection Law
jurisdiction: vn
kind: law
binding: true
status: >-
  AI Law in force 1 March 2026, with grace to 1 March 2027 (1 September 2027 for health,
  education and finance); Personal Data Protection Law in force 1 January 2026
as_of: 2026-09-24
summary: >-
  Risk-based classification of AI systems, disclosure when people interact with AI,
  labelling of AI-generated content; personal-data impact assessments, cross-border
  transfer dossiers and 72-hour breach notification.
sources:
  - https://www.loc.gov/item/global-legal-monitor/2026-04-03/vietnam-ai-law-takes-effect
  - https://iapp.org/news/a/vietnam-s-first-standalone-ai-law-an-overview-of-key-provisions-future-implications
requirements:
  - {ref: "AI Law — classification", title: "AI system classified by risk", controls: [C12]}
  - {ref: "AI Law — transparency", title: "People told they are interacting with AI", controls: [C9]}
  - {ref: "AI Law — labelling", title: "AI-generated content labelled", controls: [C9]}
  - {ref: "AI Law — high risk", title: "High-risk systems: oversight, records, incident handling", controls: [C8, C10, C15]}
  - {ref: "PDP Law — impact assessment", title: "Personal data processing impact assessment", controls: [C12]}
  - {ref: "PDP Law — transfers", title: "Cross-border transfer impact assessment dossier", controls: [C6], manual: true}
  - {ref: "PDP Law — rights", title: "Access, correction and deletion", controls: [C11]}
  - {ref: "PDP Law — breach", title: "Breach notified within 72 hours", controls: [C15]}
policy:
  transparency: {disclose: true, label_output: true, label_for: [vn], provenance: true, languages: [vi]}
incidents:
  personal_data_breach:
    - {notify: Ministry of Public Security (A05), within: 72, basis: "PDP Law — 72 hours", min_severity: medium}
"####,
    ),
];
