"""
Defines the golden test cases for evaluating the LexAI pipeline.
Each test case contains the input query and the manually verified expected output.
"""

EVAL_TEST_CASES = [
    {
        "id": "murder_punishment",
        "input": "What is the punishment for murder (qatl-e-amd) under the Pakistan Penal Code?",
        "expected_output": "Under Section 302 of the PPC, the punishment for qatl-e-amd includes death, imprisonment for life, and in some cases, imprisonment which may extend to twenty-five years as ta'zir."
    },
    {
        "id": "valid_contract",
        "input": "What are the essential requirements for a valid contract in Pakistan?",
        "expected_output": "According to Section 10 of the Contract Act, 1872, essential requirements include free consent of parties competent to contract, a lawful consideration, and a lawful object."
    },
    {
        "id": "bail_non_bailable",
        "input": "What are the conditions for granting bail in non-bailable offences under CrPC?",
        "expected_output": "Under Section 497 of the CrPC, bail in non-bailable offences is generally at the court's discretion but is restricted if there are reasonable grounds believing the accused committed an offence punishable with death or life imprisonment, with exceptions for minors, women, or sick individuals."
    },
    {
        "id": "burden_of_proof",
        "input": "Who holds the burden of proof in Pakistani courts?",
        "expected_output": "Under Article 117 of the Qanun-e-Shahadat Order, 1984, the burden of proof lies on the person who asserts the existence of any legal right or liability dependent on the facts they allege."
    },
    {
        "id": "fair_trial",
        "input": "Does a citizen have the right to a fair trial under the Constitution?",
        "expected_output": "Yes, Article 10A of the Constitution of Pakistan strictly guarantees the right to a fair trial and due process for the determination of civil rights and obligations or in any criminal charge."
    },
    {
        "id": "fraud_definition",
        "input": "How is fraud defined in the Contract Act?",
        "expected_output": "Section 17 of the Contract Act defines fraud as acts committed by a party to a contract with intent to deceive another party, including making false suggestions, active concealment of facts, or promises made without intention of performing them."
    },
    {
        "id": "theft_punishment",
        "input": "What is theft and its punishment?",
        "expected_output": "Under Section 378 of the PPC, theft is moving movable property out of someone's possession without consent. Section 379 prescribes punishment of imprisonment extending up to three years, a fine, or both."
    },
    {
        "id": "culpable_homicide",
        "input": "What is the difference between culpable homicide and murder?",
        "expected_output": "Culpable homicide is causing death with the intention or knowledge of causing death or bodily injury likely to cause death. It escalates to murder (qatl-e-amd) when specific strict conditions of intention and imminence of danger under the PPC are met."
    },
    {
        "id": "arrest_rights",
        "input": "What are fundamental rights upon arrest and detention?",
        "expected_output": "Article 10 of the Constitution mandates that an arrested person must be informed of the grounds for arrest, has the right to consult a legal practitioner, and must be produced before a magistrate within 24 hours."
    },
    {
        "id": "defamation",
        "input": "How is defamation treated under the PPC?",
        "expected_output": "Section 499 of the PPC defines defamation as making or publishing imputations to harm a person's reputation. Section 500 sets the punishment at simple imprisonment for up to two years, a fine, or both."
    }
]