from rofa.core.parse import extract_choice_letter


def test_extract_choice_letter_examples():
    samples = [
        ("Final answer: A", "A"),
        ("Answer - b", "B"),
        ("So the answer is C.", "C"),
        ("The correct answer is D", "D"),
        ("After thinking, the answer is A", "A"),
        ("Option B is correct because...", "B"),
        ("(C) is best", "C"),
        ("We choose D!", "D"),
        ("\n\nD.\n", "D"),
        ("We list options:\nA. first\nB. second\nC. third\nD. fourth\nTherefore: B", "B"),
        ("Reasoning... final answer: d", "D"),
        ("The answer is A but maybe B", "A"),
    ]

    for text, expected in samples:
        assert extract_choice_letter(text) == expected


def test_extract_choice_letter_from_options():
    options = {
        "A": "9 minutes",
        "B": "4.5 minutes",
        "C": "27 minutes",
        "D": "13.5 minutes",
    }
    text = "The final answer is 27 minutes after scaling the rate to 1/3."
    assert extract_choice_letter(text, options=options) == "C"

    list_options = ["cyan", "magenta", "yellow", "black"]
    text = "Printing uses black ink for the key color."
    assert extract_choice_letter(text, options=list_options) == "D"

    options = {
        "A": "Urinary concentration in g/24 hour",
        "B": "Urinary concentration in mg/ml",
        "C": "Urine osmolarity",
        "D": "Urine volume per minute",
    }
    text = (
        "U in the formula for urea clearance stands for urinary urea concentration, "
        "typically expressed in mg/ml."
    )
    assert extract_choice_letter(text, options=options) == "B"


def test_extract_choice_letter_priority_examples():
    samples = [
        (
            "B. Estrogen\n\nA is incorrect because ...\n"
            "Option C might seem plausible but ...\n"
            "Therefore, the correct answer is B.",
            "B",
        ),
        (
            "C. Start empiric antibiotics.\n\n"
            "Option A is not recommended in pregnancy.\n"
            "Option B is outdated.\n",
            "C",
        ),
        ("We consider options A, B, C.\nFINAL: D\n", "D"),
        ("After considering the differential diagnosis, Answer: A.\nOption B would be used only if...", "A"),
        ("Option A could work. Option B could work too.\nNot enough information provided.\n", None),
        ("A is wrong because...\nB is wrong because...\nSo the answer is: C\n", "C"),
    ]

    for text, expected in samples:
        assert extract_choice_letter(text) == expected


def test_extract_choice_letter_first_line_regression_cases():
    cases = [
        (
            "B. Estrogen\n\n"
            "Step-by-step reasoning:\n\n"
            "1. Sex hormone-binding globulin (SHBG) is a protein that binds to sex hormones...\n",
            "B",
        ),
        (
            "C. Cervical head gear\n\n"
            "Explanation: In a Class II division 1 malocclusion...\n",
            "C",
        ),
        (
            "D. Chorionic villous biopsy\n\n"
            "Chorionic villous biopsy (CVS) is a prenatal diagnostic test...\n",
            "D",
        ),
        ("A. Artesunate plus lumefantrine\n", "A"),
        ("C. b-2 microglobulin\n", "C"),
        (
            "We consider options A, B, C.\n"
            "Therefore, the correct answer is D.\n",
            "D",
        ),
    ]

    for text, expected in cases:
        assert extract_choice_letter(text) == expected


def test_extract_choice_letter_regression_cases():
    cases = [
        {
            "index": 2,
            "id": "caec1fed-b5cf-4aa1-98bd-0c0806992f06",
            "expected": "B",
            "model_output": (
                "B. Estrogen\n\nStep-by-step reasoning:\n\n"
                "1. Sex hormone-binding globulin (SHBG) is a protein that binds to sex hormones, primarily testosterone and estradiol (a form of estrogen), in the bloodstream.\n"
                "2. When SHBG levels change, it affects the free, unbound levels of these hormones, not the total hormone levels.\n"
                "3. Testosterone (option A) is primarily bound to SHBG, so changes in SHBG will affect testosterone levels. However, since the question asks for the hormone affected \"most,\" we need to consider which hormone is more significantly impacted by changes in SHBG.\n"
                "4. Estrogen (option B) is also bound to SHBG, but it has a higher affinity for albumin, another protein that binds to sex hormones. Therefore, changes in SHBG primarily affect testosterone levels more than estrogen levels.\n"
                "5. Progesterone (option C) is not significantly bound to SHBG, as it binds primarily to albumin and corticosteroid-binding globulin. Thus, changes in SHBG do not affect progesterone levels as much as testosterone levels.\n"
                "6. DHEA (option D) is a precursor hormone that is not significantly bound to SHBG. Changes in SHBG do not affect DHEA levels.\n\n"
                "Based on this reasoning, estrogen (option B) is the hormone most affected by changes in sex hormone-binding globulin, as testosterone is more significantly impacted due to its higher affinity for SHBG."
            ),
        },
        {
            "index": 7,
            "id": "4e9a8341-05b8-4b00-859b-7a13d13b81a0",
            "expected": "C",
            "model_output": (
                "C. Cervical head gear\n\n"
                "Explanation: In a Class II division 1 malocclusion with a horizontal growth trend, the cervical headgear is preferred. "
                "This type of headgear works by pulling the maxilla forward and downward, which helps to correct the Class II relationship and address the horizontal growth pattern. "
                "High pull and medium pull headgears are not suitable for this type of malocclusion, as they would exacerbate the horizontal growth trend. "
                "Reverse pull headgear is used for Class III malocclusions."
            ),
        },
        {
            "index": 16,
            "id": "1198f3e5-a21e-4487-ac57-6ab1dee93d67",
            "expected": "C",
            "model_output": (
                "C. Acute myocardial infarction\n\n"
                "Step-by-step reasoning:\n"
                "1. The patient is a smoker, which increases the risk of cardiovascular diseases, including coronary artery disease and myocardial infarction (heart attack).\n"
                "2. The patient presents with recent onset breathlessness, which could be a symptom of a heart-related issue, such as a heart attack.\n"
                "3. An ECG (electrocardiogram) was done, which is a diagnostic tool for evaluating heart rhythm and detecting signs of heart attack or other cardiac conditions.\n"
                "4. Given the patient's risk factors (smoking) and symptoms (breathlessness), and considering that an ECG was performed, the most likely diagnosis is acute myocardial infarction (heart attack)."
            ),
        },
        {
            "index": 70,
            "id": "a9eacca0-36c6-4171-996c-d7c4d75f42df",
            "expected": "C",
            "model_output": (
                "C. Inguinal exploration\n\n"
                "In this scenario, the child has an undescended testis with blind testicular vessels, which indicates that the testis has not descended into the scrotum and the blood supply to the testis is not connected to the scrotal vessels. "
                "The next step would be to perform an inguinal exploration to locate the testis and reposition it into the scrotum, if possible, or perform an orchiopexy to secure the testis in its normal position. "
                "Abdominal exploration (option A) would not be the correct choice as it would not address the inguinal region where the testis is likely to be found. "
                "Doing nothing (option B) is not an option as the child needs surgical intervention to address the undescended testis. "
                "Scrotal exploration (option D) is not suitable as the testis is impalpable, meaning it cannot be felt in the scrotum, and the blind testicular vessels indicate that it is not connected to the scrotal vessels."
            ),
        },
        {
            "index": 210,
            "id": "f7aaa747-4b20-4034-b4dd-3aa8b964327a",
            "expected": "A",
            "model_output": (
                "A. Basal Cell Carcinoma\n\n"
                "Basal cell carcinoma is the least commonly associated with lymphatic spread among the given options. "
                "It typically grows locally and invades surrounding tissues, but it rarely metastasizes through the lymphatic system. "
                "The other options, squamous cell carcinoma, malignant melanoma, and Merkel cell carcinoma, are more likely to spread through the lymphatic system."
            ),
        },
        {
            "index": 300,
            "id": "0cd0e1c4-aabe-4ac3-ae2b-83b0633cb376",
            "expected": "B",
            "model_output": (
                "B. VLDL\n\n"
                "Explanation: A person on a fat-free carbohydrate-rich diet will consume more carbohydrates than fats. "
                "The excess carbohydrates will be converted to glucose and stored as glycogen in the liver and muscles. "
                "When glycogen stores are full, the liver converts the excess glucose to triglycerides, which are then packaged into VLDL (Very Low-Density Lipoprotein) for transport to peripheral tissues. "
                "Elevated VLDL levels are associated with obesity. "
                "Chylomicrons are primarily involved in dietary fat transport, HDL (High-Density Lipoprotein) is considered \"good\" cholesterol and helps remove excess cholesterol from the bloodstream, and LDL (Low-Density Lipoprotein) is associated with \"bad\" cholesterol and can contribute to plaque buildup in arteries."
            ),
        },
        {
            "index": 361,
            "id": "b3554a03-5d0e-42ef-8033-6b3062193ae1",
            "expected": "B",
            "model_output": (
                "B. Microlaryngoscopy\n\n"
                "Explanation: In this case, Kamla, a 4-year-old child, presented with mild respiratory distress and was diagnosed with multiple juvenile papillomatosis of the larynx through laryngoscopy. "
                "The next line of management would be microlaryngoscopy, which is a surgical procedure to remove the papillomas. "
                "This is the most appropriate option as it directly addresses the underlying cause of her symptoms. "
                "Tracheostomy (A) is not indicated as it is a surgical procedure to bypass the upper airway, which is not necessary in this case. "
                "Steroids (C) may be used to reduce inflammation, but they do not address the papillomas themselves. "
                "Antibiotics (D) are not indicated as there is no evidence of infection."
            ),
        },
        {
            "index": 373,
            "id": "summary.jsonl",
            "expected": "D",
            "model_output": (
                "To determine which nerve fiber type is least susceptible to the effects of local anesthetics, we need to recall the characteristics of nerve fibers.\n\n"
                "Small, myelinated fibers (B fibers) and small unmyelinated fibers (C fibers) are more susceptible.\n"
                "Large myelinated fibers are least susceptible.\n\n"
                "Therefore, the correct answer is D.\n"
            ),
        },
        {
            "index": 21,
            "id": "71f6de26-288f-4f52-8d52-acebacb7e001",
            "expected": "C",
            "model_output": (
                "Based on the information provided, the immunofluorescence staining pattern from a kidney biopsy suggests a granular pattern, which is characteristic of immune complex deposition. "
                "Among the given options, the most probable cause is:\n\n"
                "C. Lupus Nephritis\n\n"
                "This is because lupus nephritis is an immune complex-mediated glomerulonephritis, and the granular immunofluorescence pattern is a hallmark of this condition. "
                "FSGS (A) typically shows a \"spike and dome\" pattern, PSGN (B) shows a \"lumpy bumpy\" pattern, and Goodpasture syndrome (D) shows linear staining along the glomerular basement membrane."
            ),
        },
    ]

    for case in cases:
        assert extract_choice_letter(case["model_output"]) == case["expected"]


def test_extract_choice_letter_preserves_predictions():
    cases = [
        {
            "index": 0,
            "id": "6f132291-8a44-47a8-9979-1d359c074a7c",
            "expected": "D",
            "model_output": (
                "To determine the nerve supply of the mucosa of the larynx from the epiglottis to the level of the vocal cords, we need to consider the innervation provided by the branches of the vagus nerve (cranial nerve X).\n\n"
                "1. **Superior Laryngeal Nerve (Option A):** The superior laryngeal nerve is a branch of the vagus nerve. It has two branches: the internal laryngeal nerve and the external laryngeal nerve. The internal laryngeal nerve provides sensory innervation to the mucous membrane of the larynx above the vocal cords, including the epiglottis. However, it does not innervate the area below the vocal cords.\n\n"
                "2. **External Laryngeal Nerve (Option B):** The external laryngeal nerve is another branch of the superior laryngeal nerve, primarily responsible for the motor innervation of the cricothyroid muscle, which is involved in adjusting the pitch of the voice. It does not provide sensory innervation to the mucosa of the larynx.\n\n"
                "3. **Recurrent Laryngeal Nerve (Option C):** The recurrent laryngeal nerve is also a branch of the vagus nerve. It provides both motor and sensory innervation. The sensory innervation of the mucosa below the vocal cords is provided by the recurrent laryngeal nerve. However, it is not responsible for the area above the vocal cords.\n\n"
                "4. **Internal Laryngeal and Recurrent Laryngeal (Option D):** The internal laryngeal nerve, as mentioned, supplies the mucosa above the vocal cords, including the epiglottis. The recurrent laryngeal nerve supplies the mucosa below the vocal cords. Together, they cover the entire mucosal area of the larynx from the epiglottis down to the level of the vocal cords.\n\n"
                "Given the information, the correct answer is the combination of the internal laryngeal and recurrent laryngeal nerves, as they together provide the sensory innervation to the mucosa of the larynx from the epiglottis to the level of the vocal cords.\n\n"
                "So, the answer is D."
            ),
        },
        {
            "index": 1,
            "id": "3513f98e-19a9-4dfd-8f49-1e8514c3907c",
            "expected": "B",
            "model_output": (
                "To determine the correct answer, let's break down the formula for urea clearance, which is C = U x V/P, where C stands for clearance, U for the concentration of the substance in the urine, V for the urine flow rate, and P for the plasma concentration of the substance.\n\n"
                "1. **Understanding the components of the formula:**\n"
                "   - **C (Clearance):** This is the volume of plasma from which the substance is completely removed per unit time and is expressed in units of volume per time (e.g., mL/min).\n"
                "   - **U (Urine concentration):** This represents the concentration of the substance in the urine. It's crucial to note that this concentration should be in the same units as the plasma concentration (P) to allow for accurate calculation of clearance. For urea, which is often measured in blood in mg/dL, the urine concentration would ideally be in mg/mL.\n"
                "   - **V (Urine flow rate):** This is the volume of urine produced per unit time, usually measured in mL/min.\n"
                "   - **P (Plasma concentration):** This is the concentration of the substance in the plasma, which should match the units of U for the formula to be valid.\n\n"
                "2. **Evaluating the options:**\n"
                "   - A. **Urinary concentration in g/24 hour:** This option is incorrect because the units for clearance (C) are typically much smaller (e.g., mL/min), and using g/24 hour would not be appropriate for the calculation of clearance.\n"
                "   - B. **Urinary concentration in mg/ml:** This is the correct unit for U in the context of the formula. It allows for the calculation of clearance in a manner that is consistent with the units of plasma concentration (P) and urine flow rate (V).\n"
                "   - C. **Urine osmolarity:** Osmolarity is a measure of the concentration of osmotically active particles in a solution, expressed in osmoles per liter (Osm/L). While it's related to the concentration of substances in the urine, it's not what U represents in the clearance formula.\n"
                "   - D. **Urine volume per minute:** This is actually the definition of V (urine flow rate), not U.\n\n"
                "3. **Conclusion:** Based on the explanation, U in the formula for urea clearance stands for the urinary concentration of the substance, which should be in the same units as the plasma concentration for the formula to be valid. Therefore, the correct answer is the one that matches this description.\n\n"
                "Answer: B."
            ),
        },
        {
            "index": 2,
            "id": "caec1fed-b5cf-4aa1-98bd-0c0806992f06",
            "expected": "A",
            "model_output": (
                "Sex hormone-binding globulin (SHBG) is a protein that binds to sex hormones, including testosterone and estradiol (a form of estrogen), in the blood. It plays a crucial role in regulating the levels of these hormones by binding to them and transporting them in the bloodstream, thereby influencing their bioavailability and activity.\n\n"
                "A. Testosterone: Testosterone is a primary androgen hormone, and a significant portion of it is bound to SHBG. Changes in SHBG levels can directly affect the amount of free (unbound) testosterone available in the body, which is the biologically active form. An increase in SHBG would decrease the amount of free testosterone, while a decrease in SHBG would increase the amount of free testosterone.\n\n"
                "B. Estrogen: Estrogen, particularly estradiol, is also bound to SHBG. However, the body can convert androgens (like testosterone) to estrogens through the action of the enzyme aromatase. Therefore, while changes in SHBG can affect estrogen levels indirectly, the direct impact is less pronounced compared to testosterone.\n\n"
                "C. Progesterone: Progesterone is a steroid hormone involved in the menstrual cycle and pregnancy. It is not primarily bound to SHBG. Instead, it is transported in the bloodstream bound to albumin and other proteins. Thus, changes in SHBG levels have a minimal direct effect on progesterone levels.\n\n"
                "D. DHEA (Dehydroepiandrosterone): DHEA is an androgen produced by the adrenal glands. It is not primarily bound to SHBG. DHEA can be converted to more potent androgens and estrogens in the body, but its levels are not directly influenced by SHBG.\n\n"
                "Considering the above, the hormone most directly affected by changes in sex hormone-binding globulin (SHBG) levels is testosterone, as it is significantly bound to SHBG and its bioavailability is directly influenced by the levels of this protein.\n\n"
                "Answer: A."
            ),
        },
        {
            "index": 3,
            "id": "ddd74ca8-115f-444b-a754-28567fb618b1",
            "expected": None,
            "model_output": (
                "To answer this question, let's break down the definitions related to cranial and spinal fusions:\n\n"
                "1. **Craniopagus**: This term refers to a type of conjoined twinship where the twins are joined at the head. The prefix \"crani-\" comes from the Greek word for \"skull,\" indicating that the fusion involves the cranial region.\n\n"
                "2. **Spina bifida** or **spinal fusion** would refer to a condition affecting the spine, not the head. It's a congenital defect where there is incomplete closing of the backbone and membranes around the spinal cord.\n\n"
                "3. **Thoracopagus** refers to a type of conjoined twinship where the twins are joined at the chest (thorax).\n\n"
                "Given these definitions:\n"
                "- Option A (Head and spine) is incorrect because craniopagus specifically refers to the head, not the spine.\n"
                "- Option B (Head only) is correct because craniopagus involves the fusion of the heads of conjoined twins.\n"
                "- Option C (Thorax and spine) is incorrect because this describes thoracopagus, not craniopagus.\n"
                "- Option D (Thorax only) is incorrect as it does not relate to craniopagus but rather to thoracopagus.\n\n"
                "So, the correct answer is the one that specifies the fusion of the head only."
            ),
        },
        {
            "index": 4,
            "id": "d8a284c5-175a-4439-8ef2-5ff24b41f3cc",
            "expected": "B",
            "model_output": (
                "To approach this question, let's consider the key elements provided:\n\n"
                "1. **Prolonged IV cannulation**: This is a significant risk factor for developing infections, particularly those related to the skin and soft tissues. The longer the cannula is in place, the higher the risk of introducing bacteria into the bloodstream.\n\n"
                "2. **Spike of fever**: Fever is a common systemic response to infection. In the context of IV cannulation, it suggests that the infection is likely related to the site of the cannula or the IV line itself.\n\n"
                "3. **Common pathogens in IV line infections**: The most common pathogens associated with IV line infections include Staphylococcus species, particularly those that are coagulase-negative, and gram-negative bacteria like Pseudomonas aeruginosa. Coagulase-negative staphylococci (CoNS), such as Staphylococcus epidermidis, are part of the normal skin flora and are notorious for causing infections associated with indwelling medical devices, including IV lines.\n\n"
                "- **A. Pseudomonas aeruginosa**: While Pseudomonas aeruginosa is a common cause of hospital-acquired infections, it is more commonly associated with respiratory infections, particularly in immunocompromised patients or those with cystic fibrosis, and less commonly with IV line infections compared to CoNS.\n\n"
                "- **B. Coagulase-negative staphylococcus (CoNS)**: This is the most common cause of catheter-related bloodstream infections, especially in the context of prolonged IV cannulation. CoNS are part of the normal skin flora and can easily colonize the catheter.\n\n"
                "- **C. Streptococcus agalactiae (Group B Streptococcus)**: This bacterium is more commonly associated with infections in newborns, pregnant women, and individuals with certain underlying health conditions, rather than with IV line infections in otherwise healthy adults.\n\n"
                "- **D. E. coli**: While E. coli can cause a wide range of infections, it is less commonly associated with IV line infections compared to CoNS.\n\n"
                "Given these considerations, the most likely cause of a fever in a patient with prolonged IV cannulation is an infection by Coagulase-negative staphylococcus, due to its prevalence in skin flora and its propensity to cause infections in the context of indwelling medical devices.\n\n"
                "Answer: B."
            ),
        },
        {
            "index": 5,
            "id": "defe1c85-63fe-4a16-ba10-b3749975818a",
            "expected": "A",
            "model_output": (
                "To determine the correct answer, let's analyze the anatomy of the skull, specifically focusing on the cribriform plate and its connections.\n\n"
                "The cribriform plate is a part of the ethmoid bone, which is a spongy bone located at the roof of the nasal cavity. The ethmoid bone serves as a bridge between the nasal cavity and the cranial cavity. The cribriform plate is named for its sieve-like (cribriform) appearance due to the numerous small foramina (openings) it contains. These foramina allow the passage of the olfactory nerves (cranial nerve I) from the nasal cavity to the brain, enabling the sense of smell.\n\n"
                "Now, let's evaluate the options:\n\n"
                "A. Nasal cavity and anterior cranial fossa: This option correctly identifies the cribriform plate's role as a connection between the nasal cavity and the anterior cranial fossa, where the olfactory nerves transmit sensory information.\n\n"
                "B. Nasal cavity and orbit: The orbit is the bony cavity that contains the eye and its associated structures. The cribriform plate does not connect the nasal cavity to the orbit.\n\n"
                "C. Nasal cavity and oral cavity: The oral cavity is the space inside the mouth, and it is not directly connected to the nasal cavity by the cribriform plate.\n\n"
                "D. Anterior cranial fossa and orbit: While the anterior cranial fossa is correctly identified, the connection is specifically with the nasal cavity, not the orbit.\n\n"
                "Given the information, the correct answer is the one that accurately describes the cribriform plate's connection between the nasal cavity and the anterior cranial fossa.\n\n"
                "Answer: A."
            ),
        },
        {
            "index": 6,
            "id": "be168ef1-fee0-4f80-9c60-6259e78b56b9",
            "expected": "B",
            "model_output": (
                "To answer this question, let's consider what migratory motor complexes (MMC) are and their role in the gastrointestinal (GI) tract. Migratory motor complexes are bursts of electromechanical activity that occur in the GI tract during periods of fasting. They are thought to serve several functions, including clearing the stomach and small intestine of residual contents, maintaining the integrity of the mucosal barrier, and possibly preparing the gut for the next meal by clearing out any remaining food particles and secretions.\n\n"
                "The frequency and timing of MMCs are influenced by the fasting state and the circadian rhythm. Typically, after a meal, MMCs are suppressed, and they reappear after a certain period of fasting. The interval between MMCs is generally consistent and is used as a marker of the fasting state in the gut.\n\n"
                "The most commonly cited interval for the reappearance of MMCs after a meal is about 90 minutes. This is based on studies that have monitored the electromechanical activity in the GI tract using techniques such as manometry. While there can be some variation in the exact timing, 90 minutes is the most widely accepted interval for the reappearance of MMCs after a period of eating.\n\n"
                "So, let's look at the options provided:\n"
                "A. 60 minutes - This is too short for the typical reappearance of MMCs.\n"
                "B. 90 minutes - This is the most commonly cited interval for the reappearance of MMCs.\n"
                "C. 120 minutes - This is longer than the typical interval.\n"
                "D. 150 minutes - This is also longer than the typical interval.\n\n"
                "Given the information, the correct answer is the one that aligns with the established scientific understanding of MMCs and their reappearance after a meal.\n\n"
                "Answer: B."
            ),
        },
        {
            "index": 7,
            "id": "4e9a8341-05b8-4b00-859b-7a13d13b81a0",
            "expected": "C",
            "model_output": (
                "To address a Class II division 1 malocclusion with a horizontal growth trend, we need to consider the type of headgear that will effectively counteract the excessive mandibular growth or promote maxillary growth to correct the malocclusion. Here's a brief overview of the options provided:\n\n"
                "A. High pull headgear: This type of headgear is used to apply a force to the upper teeth, pulling them backward and downward. It's more commonly used in cases where there is a need to correct a Class II malocclusion with a vertical growth pattern, as it can exacerbate a horizontal growth pattern.\n\n"
                "B. Medium pull headgear: This type of headgear is less commonly discussed in the literature and its specifics can vary, but generally, it's not the first choice for addressing a Class II division 1 malocclusion with a horizontal growth pattern.\n\n"
                "C. Cervical headgear: This type of headgear is applied to the cervical region of the head, pulling the maxilla forward and downward. It's particularly useful in cases where there is a need to correct a Class II malocclusion with a horizontal growth pattern, as it can effectively counteract the excessive mandibular growth and promote a more favorable growth pattern.\n\n"
                "D. Reverse pull headgear: This type of headgear is used to pull the mandible forward, which is not the desired effect in treating a Class II division 1 malocclusion, especially with a horizontal growth trend, as it would not correct the underlying issue of excessive mandibular growth.\n\n"
                "Considering the need to correct a Class II division 1 malocclusion with a horizontal growth trend, the most appropriate headgear would be the one that can effectively counteract the excessive mandibular growth and promote a more favorable growth pattern. Cervical headgear is specifically designed to do this by pulling the maxilla forward and downward, which can help to correct the Class II relationship and address the horizontal growth pattern.\n\n"
                "So, the answer is C."
            ),
        },
        {
            "index": 8,
            "id": "e2dd93d8-8047-4e9a-9389-e2b1781f8a35",
            "expected": "A",
            "model_output": (
                "To determine the most conspicuous sign in breast cancer, let's evaluate each option:\n\n"
                "A. Nipple retraction: This is a common symptom of breast cancer, particularly in cases where the cancer is located in the upper outer quadrant of the breast. The retraction occurs due to the pulling of the nipple and areola inward by the tumor. It's often noticeable and can be a significant indicator of malignancy.\n\n"
                "B. Peau d'orange (Skin of the orange): This term describes a condition where the skin of the breast looks like the skin of an orange, due to lymphedema (swelling) caused by a blockage in the lymphatic vessels. While it is a sign associated with breast cancer, it is not as immediately noticeable as nipple retraction and usually occurs in more advanced stages.\n\n"
                "C. Puckering: Puckering of the skin can be a sign of breast cancer, but it is less specific and can also be seen in benign conditions such as fibroadenomas or fibrocystic changes. It's not as distinctive or alarming as nipple retraction.\n\n"
                "D. Cancer en-cuirasse (Cancer en-cuirasse or \"in armor\"): This term describes a hard, thickened, and often immobile breast that resembles armor. It is a late sign of breast cancer, indicating a very advanced stage of the disease. It is not as immediately noticeable or as common as nipple retraction.\n\n"
                "Considering the options, nipple retraction is the most conspicuous and specific sign of breast cancer, often prompting patients to seek medical attention and is frequently one of the first noticeable symptoms.\n\n"
                "Answer: A."
            ),
        },
        {
            "index": 9,
            "id": "6f1fdbc9-920f-4bff-b2eb-ba300fd9cc59",
            "expected": "C",
            "model_output": (
                "To determine which of the given options is not specific to a blast injury, let's analyze each option in the context of blast injuries:\n\n"
                "A. Abrasion: An abrasion is a superficial wound caused by the removal of the superficial layers of the skin, often due to friction or rubbing against a rough surface. In the context of a blast injury, an abrasion can occur due to flying debris or being thrown against a surface, making it a possible finding in blast injuries.\n\n"
                "B. Bruise: A bruise, or contusion, is caused by blood leakage into the tissues from damaged blood vessels. In a blast, the shockwave can cause blood vessels to rupture, leading to bruising. Thus, bruising can be a result of a blast injury.\n\n"
                "C. Puncture laceration: A puncture laceration is typically caused by a sharp object penetrating the skin. While it's possible for a blast to cause a puncture wound from flying debris, such as shrapnel, the term \"puncture laceration\" more commonly suggests a wound caused by a sharp, pointed object rather than the high-pressure shockwave associated with a blast.\n\n"
                "D. Fracture: A fracture is a break in a bone. Blast injuries can cause fractures due to the high-pressure shockwave that can shatter bones. The force from the blast can also cause indirect injuries, such as fractures, from being thrown or from the shockwave's impact on the body.\n\n"
                "Considering the mechanisms of injury, a puncture laceration (C) is more specifically associated with a penetrating injury from a sharp object rather than the high-pressure shockwave and flying debris typically associated with a blast injury. While it's not impossible for a blast to cause a puncture wound, the term \"puncture laceration\" is less specific to the blast mechanism compared to the other options.\n\n"
                "So, the answer is C."
            ),
        },
        {
            "index": 10,
            "id": "3b150083-1a5e-49ef-bc36-41a06b677b32",
            "expected": "D",
            "model_output": (
                "To approach this question, let's break down the phases of clinical trials and the primary objectives of each phase:\n\n"
                "1. **Phase 1 Clinical Trials**: These are the first stage of testing in human subjects and are primarily designed to assess the safety, tolerability, pharmacokinetics, and pharmacodynamics of a drug. The main goals are to determine a safe dosage range and identify side effects.\n\n"
                "2. **Safety (A)**: While safety is a critical aspect of Phase 1 trials, it is not the only aim. Safety is a broad term that encompasses several aspects, including the identification of adverse effects, but it's not the most specific or comprehensive answer for the primary aim of Phase 1 trials.\n\n"
                "3. **Efficacy (B)**: Efficacy is the ability of a drug to produce the desired therapeutic effect. Efficacy is typically assessed in later phases of clinical trials, such as Phase 2 and 3, where the drug is tested in larger populations to see if it is effective for its intended use.\n\n"
                "4. **Dose (C)**: Determining the appropriate dose is a key part of Phase 1 trials. The goal is to find the maximum tolerated dose (MTD) and the recommended phase 2 dose (RP2D), which is the dose that is most likely to succeed in later stages of testing. This is closely related to the safety and pharmacokinetics of the drug.\n\n"
                "5. **Pharmacokinetics (D)**: Pharmacokinetics involves the study of how a drug is absorbed, distributed, metabolized, and excreted in the body. This is a critical component of Phase 1 trials, as understanding how the drug behaves in the body is essential for determining the appropriate dosage and dosing schedule.\n\n"
                "Considering the above points, while all the options are relevant to Phase 1 trials, the most comprehensive and specific aim that encompasses the primary objectives of Phase 1 trials is the assessment of **pharmacokinetics** and **safety**. However, since the question asks for a single major aim and given that pharmacokinetics is a fundamental aspect of understanding how a drug behaves in the body, which is crucial for determining safety and efficacy, the best answer would focus on the core aspect that underlies the other objectives.\n\n"
                "So, the best answer is **D. Pharmacokinetics**, with the understanding that safety is also a major concern and closely related to pharmacokinetics in Phase 1 trials.\n\n"
                "However, it's worth noting that in a more nuanced interpretation, safety (A) could be considered the overarching goal, as it encompasses the outcomes of pharmacokinetics, efficacy, and dose determination. But based on the options provided and the emphasis on a single major aim, pharmacokinetics is the most specific and direct answer.\n\n"
                "However, it's worth noting that in a more nuanced interpretation, safety (A) could be considered the overarching goal, as it encompasses the outcomes of pharmacokinetics, efficacy, and dose determination. But based on the options provided and the emphasis on a single major aim, pharmacokinetics is the most specific and direct answer."
            ),
        },
        {
            "index": 11,
            "id": "157f841a-269b-4207-825d-2a62dd8cd197",
            "expected": "A",
            "model_output": (
                "To answer this question, let's consider the role and function of the adrenal glands in the fetus and the types of steroids they produce.\n\n"
                "1. **Fetal Adrenal Glands' Role**: The adrenal glands in a fetus are known to produce several hormones, but their primary role is different from that in adults. In the fetus, the adrenal glands are responsible for producing steroid hormones that are crucial for fetal development, particularly for the maturation of the reproductive and renal systems.\n\n"
                "2. **Steroid Production in Fetal Adrenal Glands**:\n"
                "   - **DHEA-S (Dehydroepiandrosterone Sulfate)**: This is a potent androgen produced by the adrenal glands in both children and adults, but it is particularly significant in the fetus. DHEA-S is a precursor to sex hormones and is crucial for the development of the reproductive system.\n"
                "   - **Cortisol**: While cortisol is a vital steroid hormone produced by the adrenal glands in adults, its production in the fetus is not as prominent as that of DHEA-S. Cortisol is more involved in the stress response and metabolism in adults.\n"
                "   - **Corticosterone**: This is another steroid hormone produced by the adrenal glands, but it is more commonly associated with the regulation of electrolyte balance and blood pressure in adults, not as the primary product in fetal development.\n"
                "   - **Progesterone**: Although progesterone is a steroid hormone, it is primarily produced by the placenta during pregnancy, not by the fetal adrenal glands. It is crucial for maintaining the pregnancy.\n\n"
                "3. **Conclusion**: Given the roles and the primary products of the fetal adrenal glands, DHEA-S stands out as the maximum steroid produced by these glands, playing a significant role in the development of the reproductive system in the fetus.\n\n"
                "**Answer: A. DHEA-S**"
            ),
        },
        {
            "index": 12,
            "id": "4506e30a-65c0-48a5-86db-595e5084787c",
            "expected": "D",
            "model_output": (
                "To determine the drug of choice (DOC) for bacterial vaginosis (BV) in pregnancy, we need to consider the safety and efficacy of the treatment options during pregnancy.\n\n"
                "1. **Bacterial vaginosis (BV)** is a common vaginal infection caused by an imbalance of the normal vaginal flora, with an overgrowth of anaerobic bacteria. It's not a sexually transmitted infection but can increase the risk of preterm labor and other complications in pregnancy.\n\n"
                "2. **Treatment Considerations in Pregnancy:**\n"
                "   - The treatment of BV in pregnancy is crucial to prevent complications. The choice of antibiotic should be effective against the anaerobic bacteria that cause BV and should be safe for use during pregnancy.\n\n"
                "3. **Evaluating the Options:**\n"
                "   - **A. Clindamycin:** This is an antibiotic that can be used to treat BV, but it's not the first-line treatment due to concerns about resistance and its use in pregnancy, although it is generally considered safe.\n"
                "   - **B. Erythromycin:** This is a macrolide antibiotic, which is not typically used for the treatment of BV. It's more commonly used for treating infections caused by Gram-positive bacteria and certain types of atypical bacteria.\n"
                "   - **C. Rovamycin (also known as Spiramycin):** This is an antibiotic used primarily for treating toxoplasmosis in pregnancy, not BV. It's not the standard treatment for bacterial vaginosis.\n"
                "   - **D. Metronidazole:** This is the first-line treatment for bacterial vaginosis in non-pregnant women and is also considered safe and effective for use during pregnancy. It's specifically indicated for the treatment of BV and is recommended by various obstetric and gynecological guidelines.\n\n"
                "**Conclusion:** Based on the safety profile and efficacy in treating bacterial vaginosis, Metronidazole is the drug of choice for treating BV in pregnancy.\n\n"
                "**Answer: D.**"
            ),
        },
        {
            "index": 13,
            "id": "bee78f93-a09e-491f-856e-846b81e4aa5c",
            "expected": "B",
            "model_output": (
                "To answer this question, let's break down the options and understand what each acronym stands for, focusing on the context of diabetic retinopathy and visual evaluation.\n\n"
                "1. **Understanding Diabetic Retinopathy**: Diabetic retinopathy is a complication of diabetes that affects the blood vessels in the retina, the light-sensitive tissue at the back of the eye. It can lead to vision loss if not properly managed. The evaluation of diabetic retinopathy is crucial for early detection and treatment to prevent severe outcomes.\n\n"
                "2. **ETDRS**: The acronym ETDRS stands for \"Early Treatment Diabetic Retinopathy Study.\" This study was a landmark clinical trial conducted by the National Eye Institute (NEI) to determine the effectiveness of laser photocoagulation in treating diabetic retinopathy. The study's findings have been widely used to develop guidelines for the management of diabetic retinopathy.\n\n"
                "3. **Analyzing the Options**:\n"
                "   - **A. Extended Treatment for Diabetic Retinopathy Study**: This does not match the known acronym ETDRS, which is specifically about early treatment, not extended treatment.\n"
                "   - **B. Early Treatment for Diabetic Retinopathy Study**: This is the correct definition of ETDRS, aligning with the study's purpose and findings.\n"
                "   - **C. Emergency Treatment for Diabetic Retinopathy Study**: This does not match the ETDRS acronym, as the study focused on early treatment rather than emergency treatment.\n"
                "   - **D. Emerging Treatment for Diabetic Retinopathy Study**: This is also incorrect, as ETDRS was not about emerging treatments but about established early treatment methods.\n\n"
                "4. **Conclusion**: Based on the information and analysis, the correct answer is the one that aligns with the known meaning of ETDRS in the context of diabetic retinopathy treatment and evaluation.\n\n"
                "Answer: B."
            ),
        },
        {
            "index": 14,
            "id": "d75dc767-c74f-4b3d-a2bd-f5e6c082e497",
            "expected": "A",
            "model_output": (
                "To determine which nerve is most likely injured in a patient with a supracondylar humerus fracture who is unable to flex the interphalangeal joint of the thumb, we need to consider the innervation of the muscles responsible for thumb movement.\n\n"
                "1. **Median Nerve (A):** The median nerve innervates the thenar muscles, which include the opponens pollicis, abductor pollicis brevis, and flexor pollicis brevis. These muscles are crucial for thumb opposition and flexion at the metacarpophalangeal (MCP) joint, but not specifically for the interphalangeal (IP) joint of the thumb. The median nerve also innervates the flexor digitorum profundus, which affects the flexion of the fingers, not the thumb's IP joint.\n\n"
                "2. **Superficial Branch of Ulnar Nerve (B):** The superficial branch of the ulnar nerve primarily provides sensory innervation to the palmar surface of the little finger and the ulnar half of the ring finger. It does not innervate muscles responsible for thumb movement.\n\n"
                "3. **Axillary Nerve (AIN) (C):** The axillary nerve primarily innervates the deltoid and teres minor muscles, which are involved in shoulder abduction and external rotation, not in the movement of the thumb's IP joint.\n\n"
                "4. **Radial Nerve (D):** The radial nerve, specifically its deep branch, innervates the extensor muscles of the wrist and fingers, including the extensor pollicis longus and brevis, which are responsible for thumb extension. However, the radial nerve does not innervate the muscles responsible for flexion of the thumb's IP joint. The flexion of the thumb's IP joint is primarily controlled by the flexor pollicis longus, which is innervated by the anterior interosseous branch of the median nerve, but more directly relevant to the question, the flexor pollicis longus is innervated by the anterior interosseous branch of the median nerve, which is a branch of the median nerve.\n\n"
                "However, considering the specific action of flexing the interphalangeal joint of the thumb, we should note that the primary nerve involved in thumb movements is the median nerve, particularly its anterior interosseous branch for the flexor pollicis longus. But given the options and focusing strictly on the direct innervation for the action described, none of the options directly mention the anterior interosseous branch of the median nerve, which is the most precise innervation for the flexor pollicis longus. Among the given choices, the median nerve is the closest related to thumb movements, but it's essential to clarify that the specific action of the IP joint of the thumb is most directly associated with the anterior interosseous branch of the median nerve, which is not listed.\n\n"
                "Given the options and the context of the question, the best choice related to thumb movement and the closest to the correct innervation for the action described would be the median nerve, despite the nuance about the specific branch not being listed.\n\n"
                "So, the answer is A."
            ),
        },
        {
            "index": 15,
            "id": "97194c13-bb56-4e06-ac35-79f69cb41bb3",
            "expected": "D",
            "model_output": (
                "To determine the correct storage conditions for Red Blood Cells (RBCs), Platelets, and Fresh Frozen Plasma (FFP), let's examine each option based on standard medical practices and guidelines:\n\n"
                "1. **Red Blood Cells (RBCs):** RBCs are typically stored at a temperature range of 2-6°C. This temperature range slows down the metabolic processes in the cells, helping to preserve their viability and function. This is a standard practice for the storage of blood components to maintain their quality and safety for transfusion.\n\n"
                "2. **Platelets:** Platelets are more sensitive to temperature and storage conditions than RBCs. They are usually stored at room temperature (20-24°C) under agitation. This is because platelets are more prone to activation and aggregation at lower temperatures, which can lead to a decrease in their function. The agitation is crucial to prevent platelet clumping and to maintain their viability.\n\n"
                "3. **Fresh Frozen Plasma (FFP):** FFP is frozen at -30°C or lower to preserve its clotting factors and other proteins. Freezing at -30°C or below is necessary to prevent the degradation of these proteins, ensuring that FFP remains effective for transfusion purposes.\n\n"
                "Given these storage requirements:\n"
                "- Option A suggests RBCs are stored at 2-6°C, which is correct, but incorrectly places FFP at 30°C and Platelets at 20-22°C, which is not the optimal temperature for platelets.\n"
                "- Option B correctly places FFP at 2-6°C, which is incorrect for FFP, and incorrectly places RBCs at -30°C.\n"
                "- Option C suggests RBCs are stored at 20-22°C, which is incorrect for RBCs, and places FFP at -30°C, which is correct, but incorrectly places Platelets at 2-6°C.\n"
                "- Option D correctly places RBCs at 20-22°C (which is not standard but seems to be a typographical error in the context of the options given, as RBCs are typically stored at 2-6°C), correctly places FFP at -30°C, and correctly places Platelets at 2-6°C, which is the standard for platelet storage.\n\n"
                "However, based on the standard practices, none of the options perfectly match the ideal storage conditions for all three components without error. The closest to standard practices, considering the need for correction in the context of standard storage conditions, would be to acknowledge that RBCs are stored at 2-6°C, FFP at -30°C, and platelets at 20-22°C under agitation. But since the question requires a choice from the provided options and considering the closest match to standard practices with the least errors, we would have to select the option that aligns with the known storage conditions for platelets and FFP, despite the error in RBC storage temperature in the options provided.\n\n"
                "Given the options and the need to select the best available choice, the focus should be on the accurate storage conditions for platelets and FFP, which are correctly identified in one of the options. However, it's crucial to note that the standard for RBCs is 2-6°C, not 20-22°C as mentioned in the options. The correct choice, based on the standard practices and the options provided, should ideally reflect the accurate storage conditions for all components, but since that's not available, we must choose the one that aligns with the known standards for platelets and FFP, acknowledging the error in RBC storage temperature in the options.\n\n"
                "The final answer is D."
            ),
        },
        {
            "index": 16,
            "id": "1198f3e5-a21e-4487-ac57-6ab1dee93d67",
            "expected": "C",
            "model_output": (
                "To approach this question, let's consider the key elements provided: the patient is a smoker, presenting with recent onset breathlessness, and an ECG was performed. Each option should be evaluated in the context of these details:\n\n"
                "A. Atrial Fibrillation (AF): AF is a common arrhythmia characterized by rapid and irregular heart rhythms. While it can cause breathlessness, it is not typically associated with a sudden onset in a smoker without other risk factors or symptoms. It's more common in older adults and those with hypertension, heart disease, or other comorbidities.\n\n"
                "B. Paroxysmal Supraventricular Tachycardia (PSVT): PSVT is a rapid heart rate originating above the ventricles. It can cause sudden onset of palpitations, chest discomfort, and breathlessness. Smokers are not at a higher risk for PSVT specifically, but the acute onset of symptoms fits with this diagnosis. The ECG would typically show a narrow complex tachycardia.\n\n"
                "C. Acute Myocardial Infarction (AMI): AMI, commonly known as a heart attack, can present with chest pain, breathlessness, and other symptoms. Smokers are at a significantly higher risk for AMI due to the damage smoking causes to the cardiovascular system. The ECG is a critical tool in diagnosing AMI, showing signs such as ST-segment elevation or depression, Q-waves, and T-wave inversion, depending on the type of infarction.\n\n"
                "D. Multi-focal Atrial Tachycardia (MAT): MAT is a less common arrhythmia characterized by an irregular heart rate with at least three different P-wave morphologies. It's often associated with underlying lung disease, such as chronic obstructive pulmonary disease (COPD), which could be related to smoking. However, the presentation is typically more chronic and less likely to be the first presentation of a new condition in a smoker.\n\n"
                "Considering the acute onset of breathlessness in a smoker, the most concerning and immediate life-threatening condition that would also be directly related to smoking is an acute myocardial infarction. The ECG would be crucial in confirming this diagnosis, showing changes indicative of ischemia or infarction.\n\n"
                "So, the answer is C."
            ),
        },
        {
            "index": 17,
            "id": "5515fbdd-b28a-4493-914c-b8cefe901334",
            "expected": "D",
            "model_output": (
                "To determine the correct answer, let's break down the definitions related to perinatal mortality and the options provided:\n\n"
                "1. **Perinatal Mortality**: This term refers to the sum of fetal deaths (stillbirths) after 28 weeks of gestation and deaths in the first week of life (early neonatal deaths). The World Health Organization (WHO) defines perinatal mortality as the number of stillbirths and deaths in the first week of life per 1,000 births.\n\n"
                "2. **Components of Perinatal Mortality**:\n"
                "   - **Stillbirths**: These are births where the fetus is born with no signs of life at or after 28 weeks of gestation or with a birth weight of 1000 grams or more.\n"
                "   - **Early Neonatal Deaths**: These are deaths that occur during the first week of life (0-6 days).\n\n"
                "3. **Analyzing the Options**:\n"
                "   - **A. Post neonate death with weight 2.5 kg**: This refers to a death that occurs after the first week of life, which is not included in the perinatal mortality rate.\n"
                "   - **B. Early neonatal with weight 1000 grams**: While a birth weight of 1000 grams is a threshold for stillbirth, the focus here is on early neonatal deaths, which are part of perinatal mortality but the weight is not the defining factor for early neonatal deaths.\n"
                "   - **C. Abortion of < 500 gram fetus**: This refers to a miscarriage or abortion of a fetus that is less than 500 grams, which is not included in the perinatal mortality rate since it occurs before 28 weeks of gestation or before a birth weight of 1000 grams.\n"
                "   - **D. Stillbirth of fetus > 500 grams**: This option aligns with the definition of stillbirths that are included in the perinatal mortality rate, as it specifies a fetus with a birth weight greater than 500 grams, which is a common threshold used in some definitions, though the WHO specifically uses 1000 grams. However, the critical aspect here is the inclusion of stillbirths, which are a component of perinatal mortality.\n\n"
                "Given the definitions, the best match for a numerator in perinatal mortality, especially focusing on the inclusion of stillbirths which are a key component, is the stillbirth of a fetus, and the option that most closely aligns with this is the one involving a fetus > 500 grams, even though the WHO standard is 1000 grams. However, the closest and most relevant option provided, considering the context of stillbirths and the inclusion in perinatal mortality, would be the one involving a fetus > 500 grams, acknowledging that the precise WHO threshold is 1000 grams.\n\n"
                "So, the answer is D."
            ),
        },
        {
            "index": 18,
            "id": "ee3ac036-6513-4b0a-a7f3-6b158504e9bd",
            "expected": "C",
            "model_output": (
                "To determine the minimum number of lobes required to form a permanent tooth, let's consider the anatomy of teeth. Teeth are classified based on their shape and the number of lobes they have. The classification is as follows:\n\n"
                "- Incisors and canines are typically unilobed, meaning they have one lobe.\n"
                "- Premolars (bicuspids) are bilobed, meaning they have two lobes.\n"
                "- Molars are generally trilobed, meaning they have three lobes.\n\n"
                "However, the key to answering this question lies in understanding that the most complex and typical form of a tooth, especially in terms of dental anatomy and development, involves more than one lobe. The question asks for the \"minimum\" number, which might initially suggest looking for the smallest number. However, in the context of dental anatomy and the development of permanent teeth, having at least three lobes is a common and necessary feature for the proper formation and function of a tooth, especially for molars which are a key part of the dental arch and are crucial for chewing.\n\n"
                "Given the options and the understanding of dental anatomy, the minimum number of lobes required to form a tooth that is structurally and functionally significant, such as a molar, would be three. This is because having fewer lobes would not provide the necessary structure for the tooth to perform its function effectively, especially in the case of molars which are designed for grinding and crushing food.\n\n"
                "So, the answer is C. 3."
            ),
        },
        {
            "index": 19,
            "id": "1245ee8a-b80f-488c-b1a2-2ad91e1484b0",
            "expected": "B",
            "model_output": (
                "To determine the correct sequence of treatment for a patient with caries in relation to teeth 51, 52, 61, and 62, we should consider the logical and clinical steps that a dentist or dental hygienist would follow. Here's the breakdown of each step in the context of managing dental caries:\n\n"
                "1. **History Taking**: This is the initial step where the dentist gathers information about the patient's medical and dental history, including any previous treatments, allergies, and current symptoms. This helps in understanding the patient's overall health and dental needs.\n\n"
                "2. **Oral Care Assessment**: Following history taking, a thorough oral examination is conducted to assess the current state of the patient's oral health, including the extent of caries, the presence of any restorations, and the condition of the teeth and gums. This step is crucial for identifying the specific caries lesions and their severity.\n\n"
                "3. **Caries Risk Assessment**: After understanding the current oral health status, assessing the risk of future caries is essential. This involves evaluating factors such as diet, oral hygiene practices, saliva flow, and past caries experience to predict the likelihood of future caries development. This step guides the preventive measures and treatment planning.\n\n"
                "4. **Diet Counseling**: Based on the caries risk assessment, dietary advice is given to the patient. This includes recommendations on diet modification to reduce the risk of caries, such as limiting sugary foods and drinks, increasing consumption of fluoride-rich foods, and maintaining a balanced diet.\n\n"
                "5. **Fluoride Application**: Fluoride application is a preventive measure aimed at strengthening tooth enamel, making teeth more resistant to acid attacks from plaque bacteria and sugars in the mouth. It is typically applied after dietary counseling to reinforce the preventive aspect of caries management.\n\n"
                "6. **Restoration**: The final step involves the actual treatment of the caries lesions. This could range from a filling (restoration) to more extensive treatments like inlays, onlays, or even extractions in severe cases. Restorations are necessary to remove decayed tooth material, prevent further decay, and restore the tooth's function and appearance.\n\n"
                "Given these steps, the logical sequence that makes the most clinical sense is to first gather a history, assess the current oral health status, evaluate the risk of future caries, provide guidance on diet to prevent further caries, apply fluoride to prevent new caries, and finally, restore the affected teeth to their healthy state.\n\n"
                "So, the correct sequence is: History taking, oral care assessment, caries risk assessment, diet counseling, fluoride application, and restoration.\n\n"
                "The answer is B."
            ),
        },
    ]

    for case in cases:
        assert extract_choice_letter(case["model_output"]) == case["expected"]
