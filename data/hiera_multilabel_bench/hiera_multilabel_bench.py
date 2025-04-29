# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hierarchical MultiLabelBench: A Benchmark Dataset for Hierarchical Multi-Label Text Classification. Version 1.0"""

import json
import os

import datasets

WOS_CONCEPTS = {
    "level_1": [
        "Psychology",
        "biochemistry",
        "CS",
        "ECE",
        "Medical",
        "MAE",
        "Civil"
    ],
    "level_2": [
        "Weight Loss",
        "Parallel computing",
        "Prenatal development",
        "Migraine",
        "Dementia",
        "Parenting",
        "Nonverbal communication",
        "Allergies",
        "Polycythemia Vera",
        "Bipolar Disorder",
        "Irritable Bowel Syndrome",
        "Internal combustion engine",
        "Overactive Bladder",
        "State space representation",
        "Borderline personality disorder",
        "Computer vision",
        "HIV/AIDS",
        "Voltage law",
        "Skin Care",
        "Thermodynamics",
        "Human Metabolism",
        "Multiple Sclerosis",
        "Genetics",
        "Problem-solving",
        "Smoking Cessation",
        "Diabetes",
        "Rheumatoid Arthritis",
        "Attention",
        "Immunology",
        "Structured Storage",
        "Signal-flow graph",
        "Strength of materials",
        "Schizophrenia",
        "Sports Injuries",
        "Enzymology",
        "Person perception",
        "Machine learning",
        "Child abuse",
        "Smart Material",
        "Bioinformatics",
        "Children's Health",
        "Mental Health",
        "Addiction",
        "Southern blotting",
        "Idiopathic Pulmonary Fibrosis",
        "Suspension Bridge",
        "Digestive Health",
        "Parkinson's Disease",
        "Hypothyroidism",
        "Stealth Technology",
        "Kidney Health",
        "Birth Control",
        "Lorentz force law",
        "Construction Management",
        "Lymphoma",
        "Psoriatic Arthritis",
        "Healthy Sleep",
        "Geotextile",
        "Distributed computing",
        "Autism",
        "Headache",
        "Machine design",
        "Gender roles",
        "network security",
        "Menopause",
        "Remote Sensing",
        "Electrical circuits",
        "Northern blotting",
        "Alzheimer's Disease",
        "Cell biology",
        "Green Building",
        "Prejudice",
        "Depression",
        "Social cognition",
        "Hydraulics",
        "Symbolic computation",
        "Manufacturing engineering",
        "Electricity",
        "Operational amplifier",
        "Senior Health",
        "Algorithm design",
        "Computer programming",
        "Relational databases",
        "PID controller",
        "Electrical generator",
        "Ankylosing Spondylitis",
        "Cryptography",
        "Polymerase chain reaction",
        "Digital control",
        "Fluid mechanics",
        "Atopic Dermatitis",
        "Operating systems",
        "Antisocial personality disorder",
        "Water Pollution",
        "Myelofibrosis",
        "Analog signal processing",
        "Prosocial behavior",
        "Control engineering",
        "Seasonal affective disorder",
        "Cancer",
        "Molecular biology",
        "Sprains and Strains",
        "Electric motor",
        "Crohn's Disease",
        "Asthma",
        "Materials Engineering",
        "Low Testosterone",
        "Hepatitis C",
        "Osteoarthritis",
        "Hereditary Angioedema",
        "Leadership",
        "Anxiety",
        "Psoriasis",
        "Ambient Intelligence",
        "Rainwater Harvesting",
        "Emergency Contraception",
        "Data structures",
        "False memories",
        "System identification",
        "Software engineering",
        "Computer graphics",
        "computer-aided design",
        "Stress Management",
        "Eating disorders",
        "Solar Energy",
        "Image processing",
        "Fungal Infection",
        "Microcontroller",
        "Medicare",
        "Osteoporosis",
        "Atrial Fibrillation",
        "Electrical network",
        "Heart Disease",
        "Media violence"
    ]
    , "parent_childs": {
        "CS": ["Symbolic computation", "Computer vision", "Computer graphics", "Operating systems", "Machine learning",
               "Data structures", "network security", "Image processing", "Parallel computing", "Distributed computing",
               "Algorithm design", "Computer programming", "Relational databases", "Software engineering",
               "Bioinformatics", "Cryptography", "Structured Storage"],
        "Medical": ["Alzheimer's Disease", "Parkinson's Disease", "Sprains and Strains", "Cancer", "Sports Injuries",
                    "Senior Health", "Multiple Sclerosis", "Hepatitis C", "Weight Loss", "Low Testosterone",
                    "Fungal Infection", "Diabetes", "Parenting", "Birth Control", "Heart Disease", "Allergies",
                    "Menopause", "Emergency Contraception", "Skin Care", "Myelofibrosis", "Hypothyroidism", "Headache",
                    "Overactive Bladder", "Irritable Bowel Syndrome", "Polycythemia Vera", "Atrial Fibrillation",
                    "Smoking Cessation", "Lymphoma", "Asthma", "Bipolar Disorder", "Crohn's Disease",
                    "Idiopathic Pulmonary Fibrosis", "Mental Health", "Dementia", "Rheumatoid Arthritis",
                    "Osteoporosis", "Medicare", "Psoriatic Arthritis", "Addiction", "Atopic Dermatitis",
                    "Digestive Health", "Healthy Sleep", "Anxiety", "Psoriasis", "Ankylosing Spondylitis",
                    "Children's Health", "Stress Management", "HIV/AIDS", "Migraine", "Osteoarthritis",
                    "Hereditary Angioedema", "Kidney Health", "Autism"],
        "Civil": ["Green Building", "Water Pollution", "Smart Material", "Ambient Intelligence",
                  "Construction Management", "Suspension Bridge", "Geotextile", "Stealth Technology", "Solar Energy",
                  "Remote Sensing", "Rainwater Harvesting"],
        "ECE": ["Electric motor", "Digital control", "Microcontroller", "Electrical network", "Electrical generator",
                "Electricity", "Operational amplifier", "Analog signal processing", "State space representation",
                "Signal-flow graph", "Electrical circuits", "Lorentz force law", "System identification",
                "PID controller", "Voltage law", "Control engineering"],
        "biochemistry": ["Molecular biology", "Enzymology", "Southern blotting", "Northern blotting",
                         "Human Metabolism", "Polymerase chain reaction", "Immunology", "Genetics", "Cell biology"],
        "MAE": ["Fluid mechanics", "Hydraulics", "computer-aided design", "Manufacturing engineering", "Machine design",
                "Thermodynamics", "Materials Engineering", "Strength of materials", "Internal combustion engine"],
        "Psychology": ["Prenatal development", "Attention", "Eating disorders", "Borderline personality disorder",
                       "Prosocial behavior", "False memories", "Problem-solving", "Prejudice",
                       "Antisocial personality disorder", "Nonverbal communication", "Leadership", "Child abuse",
                       "Gender roles", "Depression", "Social cognition", "Seasonal affective disorder",
                       "Person perception", "Media violence", "Schizophrenia"]}}

RCV_CONCEPTS = {"level_1": ["CCAT", "ECAT", "GCAT", "MCAT"],
                "level_2": ["C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C21", "C22", "C23", "C24", "C31",
                            "C32", "C33",
                            "C34", "C41", "C42", "E11", "E12", "E13", "E14", "E21", "E31", "E41", "E51", "E61", "E71",
                            "G15", "GCRIM",
                            "GDEF", "GDIP", "GDIS", "GENT", "GENV", "GFAS", "GHEA", "GJOB", "GMIL", "GOBIT", "GODD",
                            "GPOL", "GPRO",
                            "GREL", "GSCI", "GSPO", "GTOUR", "GVIO", "GVOTE", "GWEA", "GWELF", "M11", "M12", "M13",
                            "M14"],
                "level_3": ["C151", "C152", "C171", "C172", "C173", "C174", "C181", "C182", "C183", "C311", "C312",
                            "C313", "C331",
                            "C411", "E121", "E131", "E132", "E141", "E142", "E143", "E211", "E212", "E311", "E312",
                            "E313", "E411",
                            "E511", "E512", "E513", "G151", "G152", "G153", "G154", "G155", "G156", "G157", "G158",
                            "G159", "M131",
                            "M132", "M141", "M142", "M143"], "level_4": ["C1511"], "parent_childs": {
        "CCAT": ["C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C21", "C22", "C23", "C24", "C31", "C32",
                 "C33", "C34", "C41", "C42"], "C15": ["C151", "C152"], "C151": ["C1511"],
        "C17": ["C171", "C172", "C173", "C174"], "C18": ["C181", "C182", "C183"], "C31": ["C311", "C312", "C313"],
        "C33": ["C331"], "C41": ["C411"],
        "ECAT": ["E11", "E12", "E13", "E14", "E21", "E31", "E41", "E51", "E61", "E71"], "E12": ["E121"],
        "E13": ["E131", "E132"], "E14": ["E141", "E142", "E143"], "E21": ["E211", "E212"],
        "E31": ["E311", "E312", "E313"], "E41": ["E411"], "E51": ["E511", "E512", "E513"],
        "GCAT": ["G15", "GCRIM", "GDEF", "GDIP", "GDIS", "GENT", "GENV", "GFAS", "GHEA", "GJOB", "GMIL", "GOBIT",
                 "GODD", "GPOL", "GPRO", "GREL", "GSCI", "GSPO", "GTOUR", "GVIO", "GVOTE", "GWEA", "GWELF"],
        "G15": ["G151", "G152", "G153", "G154", "G155", "G156", "G157", "G158", "G159"],
        "MCAT": ["M11", "M12", "M13", "M14"], "M13": ["M131", "M132"], "M14": ["M141", "M142", "M143"]}}

BGC_CONCEPTS = {
    "level_1": ["Fiction", "Classics", "Children\u2019sBooks", "Humor", "Teen&YoungAdult", "Nonfiction", "Poetry"],
    "level_2": ["Women\u2019sFiction", "Arts&Entertainment", "Teen&YoungAdultFantasyFiction", "Sports",
                "Teen&YoungAdultRomance", "Politics", "Teen&YoungAdultFiction", "Fantasy", "NonfictionClassics",
                "Teen&YoungAdultAction&Adventure", "Teen&YoungAdultMystery&Suspense", "Self-Improvement",
                "ScienceFiction", "Biography&Memoir", "SpiritualFiction", "Children\u2019sMiddleGradeBooks",
                "LiteraryCollections", "Health&Fitness", "Business", "LiteraryCriticism", "FictionClassics", "Pets",
                "Romance", "Mystery&Suspense", "WesternFiction", "Gothic&Horror", "PopularScience", "MilitaryFiction",
                "Parenting", "History", "ParanormalFiction", "StepIntoReading", "Teen&YoungAdultSocialIssues",
                "Teen&YoungAdultHistoricalFiction", "GraphicNovels&Manga", "Cooking", "Religion&Philosophy",
                "Reference", "Teen&YoungAdultNonfiction", "Crafts,Home&Garden", "Psychology",
                "Teen&YoungAdultScienceFiction", "Travel", "HistoricalFiction", "LiteraryFiction", "Games"],
    "level_3": ["UrbanFantasy", "PerformingArts", "Economics", "Music", "MilitaryHistory", "Inspiration&Motivation",
                "Philosophy", "Children\u2019sMiddleGradeMystery&DetectiveBooks", "U.S.History", "Bibles", "Weddings",
                "ParanormalRomance", "NewAdultRomance", "Travel:Europe", "Children\u2019sMiddleGradeSportsBooks",
                "FoodMemoir&Travel", "FairyTales", "SpaceOpera", "Children\u2019sMiddleGradeHistoricalBooks",
                "Baking&Desserts", "NoirMysteries", "Technology", "WorldHistory", "Film", "EpicFantasy",
                "DomesticPolitics", "TravelWriting", "Science", "LiteraryFigureBiographies&Memoirs",
                "Regional&EthnicCooking", "Erotica", "Diet&Nutrition", "Beauty", "WesternRomance", "CookingMethods",
                "Travel:Africa", "CyberPunk", "Arts&EntertainmentBiographies&Memoirs",
                "Children\u2019sMiddleGradeFantasy&MagicalBooks", "Wine&Beverage", "HistoricalRomance", "Management",
                "Photography", "RegencyRomance", "PersonalFinance", "Travel:Asia", "Exercise",
                "Children\u2019sMiddleGradeAction&AdventureBooks", "Religion", "Writing", "Travel:Caribbean&Mexico",
                "TestPreparation", "WorldPolitics", "PersonalGrowth", "Travel:MiddleEast", "CozyMysteries",
                "Travel:Australia&Oceania", "Crafts&Hobbies", "Art", "Marketing", "EspionageMysteries", "Language",
                "SuspenseRomance", "Travel:USA&Canada", "Suspense&Thriller", "AlternativeTherapies", "CrimeMysteries",
                "Travel:Central&SouthAmerica", "MilitaryScienceFiction", "ContemporaryFantasy", "Design", "Home&Garden",
                "Health&Reference", "ContemporaryRomance", "PoliticalFigureBiographies&Memoirs",
                "HistoricalFigureBiographies&Memoirs", "SpecialtyTravel"],
    "level_4": ["CivilWarPeriod", "AncientWorldHistory", "AsianWorldHistory", "1950\u2013PresentMilitaryHistory",
                "NorthAmericanWorldHistory", "WorldWarIMilitaryHistory", "Colonial/RevolutionaryPeriod",
                "20thCenturyU.S.History", "NativeAmericanHistory", "MiddleEasternWorldHistory",
                "LatinAmericanWorldHistory", "21stCenturyU.S.History", "AfricanWorldHistory", "19thCenturyU.S.History",
                "EuropeanWorldHistory", "WorldWarIIMilitaryHistory"], "parent_childs": {
        "Biography&Memoir": ["Arts&EntertainmentBiographies&Memoirs", "PoliticalFigureBiographies&Memoirs",
                             "HistoricalFigureBiographies&Memoirs", "LiteraryFigureBiographies&Memoirs"],
        "Nonfiction": ["Biography&Memoir", "Cooking", "Arts&Entertainment", "Business", "Crafts,Home&Garden", "Games",
                       "Health&Fitness", "History", "Parenting", "Pets", "Politics", "PopularScience", "Psychology",
                       "Reference", "Religion&Philosophy", "Self-Improvement", "Sports", "Travel"],
        "Cooking": ["Regional&EthnicCooking", "CookingMethods", "FoodMemoir&Travel", "Baking&Desserts",
                    "Wine&Beverage"],
        "Arts&Entertainment": ["Art", "Design", "Film", "Music", "PerformingArts", "Photography", "Writing"],
        "Business": ["Economics", "Management", "Marketing"],
        "Crafts,Home&Garden": ["Crafts&Hobbies", "Home&Garden", "Weddings"],
        "Health&Fitness": ["AlternativeTherapies", "Diet&Nutrition", "Exercise", "Health&Reference"],
        "History": ["MilitaryHistory", "U.S.History", "WorldHistory"],
        "MilitaryHistory": ["WorldWarIIMilitaryHistory", "WorldWarIMilitaryHistory",
                            "1950\u2013PresentMilitaryHistory"],
        "U.S.History": ["21stCenturyU.S.History", "20thCenturyU.S.History", "19thCenturyU.S.History", "CivilWarPeriod",
                        "Colonial/RevolutionaryPeriod", "NativeAmericanHistory"],
        "Politics": ["DomesticPolitics", "WorldPolitics"],
        "WorldHistory": ["AfricanWorldHistory", "AncientWorldHistory", "AsianWorldHistory", "EuropeanWorldHistory",
                         "LatinAmericanWorldHistory", "MiddleEasternWorldHistory", "NorthAmericanWorldHistory"],
        "PopularScience": ["Science", "Technology"], "Reference": ["Language", "TestPreparation"],
        "Religion&Philosophy": ["Religion", "Philosophy", "Bibles"],
        "Self-Improvement": ["Beauty", "Inspiration&Motivation", "PersonalFinance", "PersonalGrowth"],
        "Travel": ["Travel:Africa", "Travel:Asia", "Travel:Australia&Oceania", "Travel:Caribbean&Mexico",
                   "Travel:Central&SouthAmerica", "Travel:Europe", "Travel:MiddleEast", "SpecialtyTravel",
                   "TravelWriting", "Travel:USA&Canada"],
        "Fiction": ["Fantasy", "Gothic&Horror", "GraphicNovels&Manga", "HistoricalFiction", "LiteraryFiction",
                    "MilitaryFiction", "Mystery&Suspense", "ParanormalFiction", "Romance", "ScienceFiction",
                    "SpiritualFiction", "WesternFiction", "Women\u2019sFiction"],
        "Fantasy": ["ContemporaryFantasy", "EpicFantasy", "FairyTales", "UrbanFantasy"],
        "Mystery&Suspense": ["CozyMysteries", "CrimeMysteries", "EspionageMysteries", "NoirMysteries",
                             "Suspense&Thriller"],
        "Romance": ["ContemporaryRomance", "Erotica", "HistoricalRomance", "NewAdultRomance", "ParanormalRomance",
                    "RegencyRomance", "SuspenseRomance", "WesternRomance"],
        "ScienceFiction": ["CyberPunk", "MilitaryScienceFiction", "SpaceOpera"],
        "Classics": ["FictionClassics", "LiteraryCollections", "LiteraryCriticism", "NonfictionClassics"],
        "Children\u2019sBooks": ["StepIntoReading", "Children\u2019sMiddleGradeBooks"],
        "Children\u2019sMiddleGradeBooks": ["Children\u2019sMiddleGradeMystery&DetectiveBooks",
                                            "Children\u2019sMiddleGradeAction&AdventureBooks",
                                            "Children\u2019sMiddleGradeFantasy&MagicalBooks",
                                            "Children\u2019sMiddleGradeHistoricalBooks",
                                            "Children\u2019sMiddleGradeSportsBooks"],
        "Teen&YoungAdult": ["Teen&YoungAdultAction&Adventure", "Teen&YoungAdultMystery&Suspense",
                            "Teen&YoungAdultFantasyFiction", "Teen&YoungAdultNonfiction", "Teen&YoungAdultFiction",
                            "Teen&YoungAdultSocialIssues", "Teen&YoungAdultHistoricalFiction", "Teen&YoungAdultRomance",
                            "Teen&YoungAdultScienceFiction"], "Humor": [], "Poetry": []}, "parent_childs": {
        "Biography&Memoir": ["Arts&EntertainmentBiographies&Memoirs", "PoliticalFigureBiographies&Memoirs",
                             "HistoricalFigureBiographies&Memoirs", "LiteraryFigureBiographies&Memoirs"],
        "Nonfiction": ["Biography&Memoir", "Cooking", "Arts&Entertainment", "Business", "Crafts,Home&Garden", "Games",
                       "Health&Fitness", "History", "Parenting", "Pets", "Politics", "PopularScience", "Psychology",
                       "Reference", "Religion&Philosophy", "Self-Improvement", "Sports", "Travel"],
        "Cooking": ["Regional&EthnicCooking", "CookingMethods", "FoodMemoir&Travel", "Baking&Desserts",
                    "Wine&Beverage"],
        "Arts&Entertainment": ["Art", "Design", "Film", "Music", "PerformingArts", "Photography", "Writing"],
        "Business": ["Economics", "Management", "Marketing"],
        "Crafts,Home&Garden": ["Crafts&Hobbies", "Home&Garden", "Weddings"],
        "Health&Fitness": ["AlternativeTherapies", "Diet&Nutrition", "Exercise", "Health&Reference"],
        "History": ["MilitaryHistory", "U.S.History", "WorldHistory"],
        "MilitaryHistory": ["WorldWarIIMilitaryHistory", "WorldWarIMilitaryHistory",
                            "1950\u2013PresentMilitaryHistory"],
        "U.S.History": ["21stCenturyU.S.History", "20thCenturyU.S.History", "19thCenturyU.S.History", "CivilWarPeriod",
                        "Colonial/RevolutionaryPeriod", "NativeAmericanHistory"],
        "Politics": ["DomesticPolitics", "WorldPolitics"],
        "WorldHistory": ["AfricanWorldHistory", "AncientWorldHistory", "AsianWorldHistory", "EuropeanWorldHistory",
                         "LatinAmericanWorldHistory", "MiddleEasternWorldHistory", "NorthAmericanWorldHistory"],
        "PopularScience": ["Science", "Technology"], "Reference": ["Language", "TestPreparation"],
        "Religion&Philosophy": ["Religion", "Philosophy", "Bibles"],
        "Self-Improvement": ["Beauty", "Inspiration&Motivation", "PersonalFinance", "PersonalGrowth"],
        "Travel": ["Travel:Africa", "Travel:Asia", "Travel:Australia&Oceania", "Travel:Caribbean&Mexico",
                   "Travel:Central&SouthAmerica", "Travel:Europe", "Travel:MiddleEast", "SpecialtyTravel",
                   "TravelWriting", "Travel:USA&Canada"],
        "Fiction": ["Fantasy", "Gothic&Horror", "GraphicNovels&Manga", "HistoricalFiction", "LiteraryFiction",
                    "MilitaryFiction", "Mystery&Suspense", "ParanormalFiction", "Romance", "ScienceFiction",
                    "SpiritualFiction", "WesternFiction", "Women\u2019sFiction"],
        "Fantasy": ["ContemporaryFantasy", "EpicFantasy", "FairyTales", "UrbanFantasy"],
        "Mystery&Suspense": ["CozyMysteries", "CrimeMysteries", "EspionageMysteries", "NoirMysteries",
                             "Suspense&Thriller"],
        "Romance": ["ContemporaryRomance", "Erotica", "HistoricalRomance", "NewAdultRomance", "ParanormalRomance",
                    "RegencyRomance", "SuspenseRomance", "WesternRomance"],
        "ScienceFiction": ["CyberPunk", "MilitaryScienceFiction", "SpaceOpera"],
        "Classics": ["FictionClassics", "LiteraryCollections", "LiteraryCriticism", "NonfictionClassics"],
        "Children\u2019sBooks": ["StepIntoReading", "Children\u2019sMiddleGradeBooks"],
        "Children\u2019sMiddleGradeBooks": ["Children\u2019sMiddleGradeMystery&DetectiveBooks",
                                            "Children\u2019sMiddleGradeAction&AdventureBooks",
                                            "Children\u2019sMiddleGradeFantasy&MagicalBooks",
                                            "Children\u2019sMiddleGradeHistoricalBooks",
                                            "Children\u2019sMiddleGradeSportsBooks"],
        "Teen&YoungAdult": ["Teen&YoungAdultAction&Adventure", "Teen&YoungAdultMystery&Suspense",
                            "Teen&YoungAdultFantasyFiction", "Teen&YoungAdultNonfiction", "Teen&YoungAdultFiction",
                            "Teen&YoungAdultSocialIssues", "Teen&YoungAdultHistoricalFiction", "Teen&YoungAdultRomance",
                            "Teen&YoungAdultScienceFiction"], "Humor": [], "Poetry": []}}

AAPD_CONCEPTS = {
    "level_1": sorted(['cs', 'physics', 'math', 'cmp-lg', 'quant-ph', 'cond-mat', 'stat', 'nlin', 'q-bio']),
    "level_2": sorted([
        'cs.pf', 'cs.cv', 'cs.ro', 'cs.sc', 'cs.hc', 'cs.ni', 'cs.lg', 'cs.ir',
        'cs.ms', 'cs.ce', 'cs.ne', 'cs.sy', 'cs.dl', 'cs.dm', 'cs.cc', 'cs.ai',
        'cs.pl', 'cs.lo', 'cs.gt', 'cs.cy', 'cs.si', 'cs.ds', 'cs.cg', 'cs.dc',
        'cs.db', 'cs.cl', 'cs.it', 'cs.fl', 'cs.cr', 'cs.na', 'cs.mm', 'cs.ma',
        'cs.se', 'math.st', 'math.pr', 'math.lo', 'math.na', 'math.oc', 'math.it',
        'math.nt', 'math.co', 'stat.me', 'stat.th', 'stat.ml', 'stat.ap',
        'physics.soc-ph', 'physics.data-an', 'cond-mat.stat-mech', 'cond-mat.dis-nn',
        'q-bio.qm', 'q-bio.nc', 'nlin.ao'
    ])
    , "parent_childs": {
        'cs': ['cs.pf', 'cs.cv', 'cs.ro', 'cs.sc', 'cs.hc', 'cs.ni', 'cs.lg', 'cs.ir', 'cs.ms', 'cs.ce', 'cs.ne',
               'cs.sy',
               'cs.dl', 'cs.dm', 'cs.cc', 'cs.ai', 'cs.pl', 'cs.lo', 'cs.gt', 'cs.cy', 'cs.si', 'cs.ds', 'cs.cg',
               'cs.dc',
               'cs.db', 'cs.cl', 'cs.it', 'cs.fl', 'cs.cr', 'cs.na', 'cs.mm', 'cs.ma', 'cs.se'],
        'physics': ['physics.soc-ph', 'physics.data-an'],
        'math': ['math.st', 'math.pr', 'math.lo', 'math.na', 'math.oc', 'math.it', 'math.nt', 'math.co'], 'cmp-lg': [],
        'quant-ph': [], 'cond-mat': ['cond-mat.stat-mech', 'cond-mat.dis-nn'],
        'stat': ['stat.me', 'stat.th', 'stat.ml', 'stat.ap'], 'nlin': ['nlin.ao'], 'q-bio': ['q-bio.qm', 'q-bio.nc']}}


MANIFESTO_CONCEPTS = {
    "level_1": sorted(['Economy',
                        'Social Groups',
                        'Freedom and Democracy',
                        'External Relations',
                        'Welfare and Quality of Life',
                        'Fabric of Society',
                        'Political System']),
    "level_2": sorted(['Technology and Infrastructure: Positive',
                        'Economic Planning',
                        'Economic Goals',
                        'Keynesian Demand Management',
                        'Foreign Special Relationships: Negative',
                        'Multiculturalism: Negative',
                        'Internationalism: Negative',
                        'Democracy',
                        'Non-economic Demographic Groups',
                        'European Community/Union: Positive',
                        'Economic Growth: Positive',
                        'Traditional Morality: Positive',
                        'Incentives: Positive',
                        'National Way of Life: Negative',
                        'Culture: Positive',
                        'Traditional Morality: Negative',
                        'Education Expansion',
                        'Anti-Imperialism',
                        'Market Regulation',
                        'Constitutionalism: Negative',
                        'Protectionism: Negative',
                        'Equality: Positive',
                        'Economic Orthodoxy',
                        'Education Limitation',
                        'Law and Order',
                        'Labour Groups: Positive',
                        'Corporatism/Mixed Economy',
                        'Peace',
                        'Anti-Growth Economy',
                        'Constitutionalism: Positive',
                        'Military: Positive',
                        'Labour Groups: Negative',
                        'Environmental Protection',
                        'Civic Mindedness: Positive',
                        'Foreign Special Relationships: Positive',
                        'Centralisation',
                        'Free Market Economy',
                        'Welfare State Expansion',
                        'National Way of Life: Positive',
                        'Nationalisation',
                        'Protectionism: Positive',
                        'Freedom and Human Rights',
                        'Decentralization',
                        'European Community/Union: Negative',
                        'Agriculture and Farmers',
                        'Political Authority',
                        'Political Corruption',
                        'Multiculturalism: Positive',
                        'Controlled Economy',
                        'Governmental and Administrative Efficiency',
                        'Internationalism: Positive',
                        'Middle Class and Professional Groups',
                        'Welfare State Limitation',
                        'Military: Negative',
                        'Marxist Analysis',
                        'Underprivileged Minority Groups']),
    "level_3": sorted(['Transition: Pre-Democratic Elites: Negative',
                        'National Way of Life General: Negative',
                        'National Way of Life: Immigration: Positive',
                        'Representative Democracy: Positive',
                        'National Way of Life: Immigration: Negative',
                        'Multiculturalism: Immigrants Diversity',
                        'Civic Mindedness General: Positive',
                        'Multiculturalism General: Positive',
                        'Law and Order: Positive',
                        'Anti-Growth Economy: Positive',
                        'Democracy General: Negative',
                        'Freedom',
                        'Agriculture and Farmers: Negative',
                        'Multiculturalism: Indigenous Rights: Positive',
                        'Political Authority: Personal Competence',
                        'National Way of Life General: Positive',
                        'Anti-Imperialism: State Centred Anti-Imperialism',
                        'Multiculturalism: Indigenous Rights: Negative',
                        'Democracy General: Positive',
                        'Law and Order: Negative',
                        'Agriculture and Farmers: Positive',
                        'Multiculturalism General: Negative',
                        'Human Rights',
                        'Transition: Rehabilitation and Compensation',
                        'Direct Democracy: Positive',
                        'Transition: Pre-Democratic Elites: Positive',
                        'Political Authority: Strong Government',
                        'Multiculturalism: Immigrants Assimilation',
                        'Sustainability: Positive',
                        'Civic Mindedness: Bottom-Up Activism',
                        'Political Authority: Party Competence',
                        'Anti-Imperialism: Foreign Financial Influence']),
    "parent_childs": {'External Relations': ['European Community/Union: Negative',
                                            'Anti-Imperialism',
                                            'Foreign Special Relationships: Positive',
                                            'Foreign Special Relationships: Negative',
                                            'Internationalism: Negative',
                                            'European Community/Union: Positive',
                                            'Peace',
                                            'Internationalism: Positive',
                                            'Military: Negative',
                                            'Military: Positive'],
                        'Freedom and Democracy': ['Freedom and Human Rights',
                                            'Constitutionalism: Negative',
                                            'Constitutionalism: Positive',
                                            'Democracy'],
                        'Political System': ['Governmental and Administrative Efficiency',
                                            'Political Authority',
                                            'Political Corruption',
                                            'Centralisation',
                                            'Decentralization'],
                        'Economy': ['Technology and Infrastructure: Positive',
                                            'Economic Planning',
                                            'Economic Goals',
                                            'Keynesian Demand Management',
                                            'Free Market Economy',
                                            'Nationalisation',
                                            'Protectionism: Positive',
                                            'Economic Growth: Positive',
                                            'Incentives: Positive',
                                            'Market Regulation',
                                            'Protectionism: Negative',
                                            'Economic Orthodoxy',
                                            'Controlled Economy',
                                            'Corporatism/Mixed Economy',
                                            'Anti-Growth Economy',
                                            'Marxist Analysis'],
                        'Welfare and Quality of Life': ['Environmental Protection',
                                            'Education Expansion',
                                            'Welfare State Limitation',
                                            'Equality: Positive',
                                            'Welfare State Expansion',
                                            'Education Limitation',
                                            'Culture: Positive'],
                        'Fabric of Society': ['Multiculturalism: Negative',
                                            'Civic Mindedness: Positive',
                                            'Multiculturalism: Positive',
                                            'Law and Order',
                                            'National Way of Life: Positive',
                                            'Traditional Morality: Positive',
                                            'Traditional Morality: Negative',
                                            'National Way of Life: Negative'],
                        'Social Groups': ['Agriculture and Farmers: Positive',
                                            'Labour Groups: Positive',
                                            'Underprivileged Minority Groups',
                                            'Middle Class and Professional Groups',
                                            'Non-economic Demographic Groups',
                                            'Labour Groups: Negative'],
                        'Anti-Imperialism': ['Anti-Imperialism: State Centred Anti-Imperialism',
                                            'Anti-Imperialism: Foreign Financial Influence'],
                        'Freedom and Human Rights': ['Freedom', 'Human Rights'],
                        'Democracy': ['Democracy General: Negative',
                                            'Democracy General: Positive',
                                            'Representative Democracy: Positive',
                                            'Direct Democracy: Positive'],
                        'Political Authority': ['Transition: Pre-Democratic Elites: Positive',
                                            'Political Authority: Strong Government',
                                            'Political Authority: Party Competence',
                                            'Political Authority: Personal Competence',
                                            'Transition: Pre-Democratic Elites: Negative',
                                            'Transition: Rehabilitation and Compensation'],
                        'Anti-Growth Economy': ['Anti-Growth Economy: Positive',
                                            'Sustainability: Positive'],
                        'National Way of Life: Positive': ['National Way of Life: Immigration: Negative',
                                            'National Way of Life General: Positive'],
                        'National Way of Life: Negative': ['National Way of Life: Immigration: Positive',
                                            'National Way of Life General: Negative'],
                        'Law and Order': ['Law and Order: Negative',
                                            'Law and Order: Positive'],
                        'Civic Mindedness: Positive': ['Civic Mindedness General: Positive',
                                            'Civic Mindedness: Bottom-Up Activism'],
                        'Multiculturalism: Positive': ['Multiculturalism: Immigrants Diversity',
                                            'Multiculturalism General: Positive',
                                            'Multiculturalism: Indigenous Rights: Positive'],
                        'Multiculturalism: Negative': ['Multiculturalism General: Negative',
                                            'Multiculturalism: Immigrants Assimilation',
                                            'Multiculturalism: Indigenous Rights: Negative'],
                        'Agriculture and Farmers': ['Agriculture and Farmers: Positive',
                                            'Agriculture and Farmers: Negative']}
}


WOS_CONCEPTS["level_all"] = sorted(WOS_CONCEPTS['level_1']) + sorted(WOS_CONCEPTS['level_2'])
AAPD_CONCEPTS["level_all"] = sorted(AAPD_CONCEPTS['level_1']) + sorted(AAPD_CONCEPTS['level_2'])
RCV_CONCEPTS["level_all"] = sorted(RCV_CONCEPTS['level_1']) + sorted(RCV_CONCEPTS['level_2']) + sorted(
    RCV_CONCEPTS['level_3']) + sorted(RCV_CONCEPTS['level_4'])
BGC_CONCEPTS["level_all"] = sorted(BGC_CONCEPTS['level_1']) + sorted(BGC_CONCEPTS['level_2']) + sorted(
    BGC_CONCEPTS['level_3']) + sorted(BGC_CONCEPTS['level_4'])
MANIFESTO_CONCEPTS["level_all"] = sorted(MANIFESTO_CONCEPTS['level_1']) + sorted(MANIFESTO_CONCEPTS['level_2']) + sorted(
    MANIFESTO_CONCEPTS['level_3'])
MAIN_PATH = 'data_files'


class HieraMultiLabelBenchConfig(datasets.BuilderConfig):
    """BuilderConfig for HieraMultiLabelBench."""

    def __init__(
            self,
            text_column,
            label_column,
            url,
            data_url,
            data_file,
            citation,
            label_level,
            label_classes=None,
            dev_column="dev",
            **kwargs,
    ):
        """BuilderConfig for LexGLUE.

        Args:
          text_column: ``string`, name of the column in the jsonl file corresponding
            to the text
          label_column: `string`, name of the column in the jsonl file corresponding
            to the label
          url: `string`, url for the original project
          data_url: `string`, url to download the zip file from
          data_file: `string`, filename for data set
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          label_classes: `list[string]`, the list of classes if the label is
            categorical. If not provided, then the label will be of type
            `datasets.Value('float32')`.
          multi_label: `boolean`, True if the task is multi-label
          dev_column: `string`, name for the development subset
          **kwargs: keyword arguments forwarded to super.
        """
        super(HieraMultiLabelBenchConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.text_column = text_column
        self.label_column = label_column
        self.label_level = label_level
        self.label_classes = label_classes
        self.dev_column = dev_column
        self.url = url
        self.data_url = data_url
        self.data_file = data_file
        self.citation = citation


class HieraMultiLabelBench(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        HieraMultiLabelBenchConfig(
            name="wos-all",
            description="web of science",
            text_column="text",
            label_column="wos_concepts",
            label_level='level_all',
            label_classes=WOS_CONCEPTS["level_all"],
            dev_column="dev",
            url='',
            data_url=f"wos.tar.gz",
            data_file="wos.jsonl",
            citation='',
        ),

        HieraMultiLabelBenchConfig(
            name="rcv-all",
            description="reuters",
            text_column="text",
            label_column="rcv_concepts",
            label_level='level_all',
            label_classes=RCV_CONCEPTS["level_all"],
            dev_column="dev",
            url='',
            data_url=f"rcv.tar.gz",
            data_file="rcv.jsonl",
            citation='',
        ),
        HieraMultiLabelBenchConfig(
            name="bgc-all",
            description="blurb genre collection",
            text_column="text",
            label_column="bgc_concepts",
            label_level='level_all',
            label_classes=BGC_CONCEPTS["level_all"],
            dev_column="dev",
            url='',
            data_url=f"bgc.tar.gz",
            data_file="bgc.jsonl",
            citation='',
        )
        ,
        HieraMultiLabelBenchConfig(
            name="aapd-all",
            description="aapd",
            text_column="text",
            label_column="aapd_concepts",
            label_level='level_all',
            label_classes=AAPD_CONCEPTS["level_all"],
            dev_column="dev",
            url='',
            data_url=f"aapd.tar.gz",
            data_file="aapd.jsonl",
            citation='',
        ),

        HieraMultiLabelBenchConfig(
            name="manifesto-all",
            description="manifesto coding",
            text_column="text",
            label_column="manifesto_concepts",
            label_level='level_all',
            label_classes=MANIFESTO_CONCEPTS["level_all"],
            dev_column="dev",
            url='',
            data_url=f"manifesto.tar.gz",
            data_file="manifesto.jsonl",
            citation='',
        )

    ]

    def _info(self):
        features = {"text": datasets.Value("string"),
                    "concepts": datasets.features.Sequence(datasets.ClassLabel(names=self.config.label_classes))}

        return datasets.DatasetInfo(
            description=self.config.description,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(os.path.join(MAIN_PATH, self.config.data_url))
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, self.config.data_file), "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": os.path.join(data_dir, self.config.data_file), "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, self.config.data_file),
                    "split": self.config.dev_column,
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """This function returns the examples in the raw (text) form."""
        with open(filepath, "r", encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                if data["data_type"] == split:
                    if "_all" not in self.config.label_level:
                        yield id_, {
                            "text": data[self.config.text_column],
                            "concepts": sorted(data[self.config.label_column][self.config.label_level]),
                        }

                    else:

                        if "aapd" in self.config.name:
                            labels = data[self.config.label_column]
                            yield id_, {
                                "text": data[self.config.text_column],
                                "concepts": sorted(labels["level_1"]) + sorted(labels["level_2"]),
                            }
                        elif "wos" in self.config.name:
                            labels = data[self.config.label_column]
                            yield id_, {
                                "text":  data[self.config.text_column],
                                "concepts": sorted(labels["level_1"]) + sorted(labels["level_2"]),
                            }
                        elif "manifesto" in self.config.name:
                            labels = data[self.config.label_column]
                            yield id_, {
                                "text":  data[self.config.text_column],
                                "concepts": sorted(labels["level_1"]) + sorted(labels["level_2"]) + sorted(labels["level_3"]),
                            }
                        else:
                            labels = data[self.config.label_column]
                            yield id_, {
                                "text": data[self.config.text_column],
                                "concepts": sorted(labels['level_1']) + sorted(labels['level_2']) + sorted(
                                    labels['level_3']) + sorted(labels['level_4']),
                            }
