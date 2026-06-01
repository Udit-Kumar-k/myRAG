import os
import json
import time
from typing import List, Dict, Any

# Curated 60-query evaluation dataset
EVAL_QUERIES = [
    # --- Category 1: Single-country factual (20 queries) ---
    {
        "id": "single_01",
        "category": "single_country",
        "question": "What is India's 2030 emissions intensity reduction target?",
        "geography_iso": "IND",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["45 percent", "intensity", "2030", "2005"]
    },
    {
        "id": "single_02",
        "category": "single_country",
        "question": "What emission reduction target did the United States set in its 2021 NDC?",
        "geography_iso": "USA",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["50-52 percent", "50 to 52 percent", "2030", "2005"]
    },
    {
        "id": "single_03",
        "category": "single_country",
        "question": "What is China's target year for achieving carbon neutrality?",
        "geography_iso": "CHN",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["carbon neutrality", "neutrality", "before 2060", "2060"]
    },
    {
        "id": "single_04",
        "category": "single_country",
        "question": "What is Brazil's 2030 emissions reduction target relative to 2005 levels?",
        "geography_iso": "BRA",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["50 percent", "50%", "2030", "2005"]
    },
    {
        "id": "single_05",
        "category": "single_country",
        "question": "Under Canada's Net-Zero Emissions Accountability Act, what is the target for 2050?",
        "geography_iso": "CAN",
        "expected_namespace": "national_laws",
        "ground_truth_keywords": ["net-zero", "net zero emissions", "2050"]
    },
    {
        "id": "single_06",
        "category": "single_country",
        "question": "What is the UK's legally binding net-zero greenhouse gas emissions target year?",
        "geography_iso": "GBR",
        "expected_namespace": "national_laws",
        "ground_truth_keywords": ["net zero", "net-zero", "2050", "Climate Change Act"]
    },
    {
        "id": "single_07",
        "category": "single_country",
        "question": "What emissions reduction target does Japan have for fiscal year 2030?",
        "geography_iso": "JPN",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["46 percent", "46%", "2030", "fiscal year 2013"]
    },
    {
        "id": "single_08",
        "category": "single_country",
        "question": "What is Germany's greenhouse gas reduction target for 2030 under its Climate Action Law?",
        "geography_iso": "DEU",
        "expected_namespace": "national_laws",
        "ground_truth_keywords": ["65 percent", "65%", "2030", "Climate Protection Act"]
    },
    {
        "id": "single_09",
        "category": "single_country",
        "question": "What is South Africa's target emissions range for 2030 in its updated NDC?",
        "geography_iso": "ZAF",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["350 to 420", "350-420 Mt CO2-eq", "2030"]
    },
    {
        "id": "single_10",
        "category": "single_country",
        "question": "What is South Korea's 2030 NDC emissions reduction target?",
        "geography_iso": "KOR",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["40 percent", "40%", "2030", "2018"]
    },
    {
        "id": "single_11",
        "category": "single_country",
        "question": "What is Turkey's emissions reduction target by 2030 in its NDC?",
        "geography_iso": "TUR",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["41 percent", "41%", "2030", "business-as-usual"]
    },
    {
        "id": "single_12",
        "category": "single_country",
        "question": "What is Indonesia's unconditional emissions reduction target in its NDC?",
        "geography_iso": "IDN",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["31.89 percent", "31.89%", "29 percent", "2030"]
    },
    {
        "id": "single_13",
        "category": "single_country",
        "question": "What does Mexico's General Climate Change Law specify for renewable energy targets?",
        "geography_iso": "MEX",
        "expected_namespace": "national_laws",
        "ground_truth_keywords": ["35 percent", "35%", "clean energy", "renewable"]
    },
    {
        "id": "single_14",
        "category": "single_country",
        "question": "What is Argentina's absolute emissions cap for 2030 under its NDC?",
        "geography_iso": "ARG",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["349 MtCO2e", "349 Mt", "absolute limit", "2030"]
    },
    {
        "id": "single_15",
        "category": "single_country",
        "question": "What does Saudi Arabia's NDC state regarding its annual greenhouse gas reduction target by 2030?",
        "geography_iso": "SAU",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["278 million tons", "278 Mt", "2030", "annually"]
    },
    {
        "id": "single_16",
        "category": "single_country",
        "question": "What is Russia's greenhouse gas emission limit for 2030 in its NDC?",
        "geography_iso": "RUS",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["70 percent", "70%", "1990 levels", "2030"]
    },
    {
        "id": "single_17",
        "category": "single_country",
        "question": "What is the European Union's legally binding emissions reduction target for 2030?",
        "geography_iso": "EU",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["55 percent", "55%", "at least 55%", "2030"]
    },
    {
        "id": "single_18",
        "category": "single_country",
        "question": "What net-zero target year is specified in France's Energy and Climate Law?",
        "geography_iso": "FRA",
        "expected_namespace": "national_laws",
        "ground_truth_keywords": ["carbon neutrality", "neutrality", "2050"]
    },
    {
        "id": "single_19",
        "category": "single_country",
        "question": "What does Australia's Climate Change Act 2022 mandate for 2030 emissions reductions?",
        "geography_iso": "AUS",
        "expected_namespace": "national_laws",
        "ground_truth_keywords": ["43 percent", "43%", "2030", "2005 levels"]
    },
    {
        "id": "single_20",
        "category": "single_country",
        "question": "What does Italy's Climate Decree specify for environmental recovery funds?",
        "geography_iso": "ITA",
        "expected_namespace": "national_laws",
        "ground_truth_keywords": ["funding", "recovery", "investments", "climate"]
    },

    # --- Category 2: Multi-country comparison (20 queries) ---
    {
        "id": "multi_01",
        "category": "multi_country",
        "question": "Which country has a stronger 2030 emission reduction target relative to 2005 levels — EU or USA?",
        "geography_iso": "USA", # requires both EU and USA context
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["EU", "55%", "USA", "50-52%", "2005"]
    },
    {
        "id": "multi_02",
        "category": "multi_country",
        "question": "Compare the carbon neutrality target years of China and India.",
        "geography_iso": "CHN", # requires CHN and IND
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["China", "2060", "India", "2070", "carbon neutrality"]
    },
    {
        "id": "multi_03",
        "category": "multi_country",
        "question": "How do the 2030 emission reduction targets of Canada and Australia compare in base years?",
        "geography_iso": "CAN",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["Canada", "40-45%", "Australia", "43%", "2005"]
    },
    {
        "id": "multi_04",
        "category": "multi_country",
        "question": "Compare the 2030 renewable energy target shares of Germany and France.",
        "geography_iso": "DEU",
        "expected_namespace": "national_laws",
        "ground_truth_keywords": ["Germany", "France", "renewable", "target"]
    },
    {
        "id": "multi_05",
        "category": "multi_country",
        "question": "How do the forest protection and deforestation targets of Brazil and Indonesia compare?",
        "geography_iso": "BRA",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["Brazil", "Indonesia", "deforestation", "forest"]
    },
    {
        "id": "multi_06",
        "category": "multi_country",
        "question": "Compare the coal phase-out commitments of the United Kingdom and Germany.",
        "geography_iso": "GBR",
        "expected_namespace": "national_laws",
        "ground_truth_keywords": ["UK", "Germany", "coal", "phase-out"]
    },
    {
        "id": "multi_07",
        "category": "multi_country",
        "question": "Which country has an earlier target for net-zero emissions between Japan and South Korea?",
        "geography_iso": "JPN",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["Japan", "South Korea", "2050", "net-zero"]
    },
    {
        "id": "multi_08",
        "category": "multi_country",
        "question": "Compare the 2030 emission intensity targets of India and China.",
        "geography_iso": "IND",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["India", "China", "intensity", "2030"]
    },
    {
        "id": "multi_09",
        "category": "multi_country",
        "question": "How do the absolute emission targets of Russia and Argentina for 2030 compare?",
        "geography_iso": "RUS",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["Russia", "Argentina", "emissions", "2030"]
    },
    {
        "id": "multi_10",
        "category": "multi_country",
        "question": "Compare the carbon pricing or carbon tax legislation in Canada and Mexico.",
        "geography_iso": "CAN",
        "expected_namespace": "national_laws",
        "ground_truth_keywords": ["Canada", "Mexico", "carbon tax", "pricing"]
    },
    {
        "id": "multi_11",
        "category": "multi_country",
        "question": "Compare the methane emission reduction pledges of the USA and Brazil.",
        "geography_iso": "USA",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["USA", "Brazil", "methane", "reduction"]
    },
    {
        "id": "multi_12",
        "category": "multi_country",
        "question": "What are the differences in 2030 emission baseline years between the EU, USA, and Japan?",
        "geography_iso": "EU",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["EU", "USA", "Japan", "baseline", "2005", "1990", "2013"]
    },
    {
        "id": "multi_13",
        "category": "multi_country",
        "question": "Compare the electric vehicle targets in domestic policies of the UK and France.",
        "geography_iso": "GBR",
        "expected_namespace": "national_laws",
        "ground_truth_keywords": ["UK", "France", "electric vehicle", "EV"]
    },
    {
        "id": "multi_14",
        "category": "multi_country",
        "question": "Contrast the carbon neutrality deadlines of Saudi Arabia and South Africa.",
        "geography_iso": "SAU",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["Saudi Arabia", "South Africa", "2060", "2050", "neutrality"]
    },
    {
        "id": "multi_15",
        "category": "multi_country",
        "question": "How do the climate adaptation strategies in the NDCs of Indonesia and Turkey compare?",
        "geography_iso": "IDN",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["Indonesia", "Turkey", "adaptation", "resilience"]
    },
    {
        "id": "multi_16",
        "category": "multi_country",
        "question": "Compare the renewable electricity capacity targets of India and Germany by 2030.",
        "geography_iso": "IND",
        "expected_namespace": "all",
        "ground_truth_keywords": ["India", "Germany", "renewable", "electricity", "GW"]
    },
    {
        "id": "multi_17",
        "category": "multi_country",
        "question": "Compare the offshore wind power generation targets of the UK and China.",
        "geography_iso": "GBR",
        "expected_namespace": "all",
        "ground_truth_keywords": ["UK", "China", "wind", "offshore"]
    },
    {
        "id": "multi_18",
        "category": "multi_country",
        "question": "Contrast the carbon capture, utilization, and storage (CCUS) policies of the USA and Saudi Arabia.",
        "geography_iso": "USA",
        "expected_namespace": "all",
        "ground_truth_keywords": ["USA", "Saudi Arabia", "CCUS", "carbon capture"]
    },
    {
        "id": "multi_19",
        "category": "multi_country",
        "question": "Compare the emission reduction commitments for the agricultural sector in the NDCs of Australia and Canada.",
        "geography_iso": "AUS",
        "expected_namespace": "ndc_commitments",
        "ground_truth_keywords": ["Australia", "Canada", "agriculture", "emissions"]
    },
    {
        "id": "multi_20",
        "category": "multi_country",
        "question": "How do the energy efficiency targets of Japan and Italy compare in their national plans?",
        "geography_iso": "JPN",
        "expected_namespace": "national_laws",
        "ground_truth_keywords": ["Japan", "Italy", "energy efficiency", "efficiency"]
    },

    # --- Category 3: Law vs pledge (20 queries) ---
    {
        "id": "law_vs_pledge_01",
        "category": "law_vs_pledge",
        "question": "Does Germany's Climate Protection Act match its NDC greenhouse gas commitments?",
        "geography_iso": "DEU",
        "expected_namespace": "all",
        "ground_truth_keywords": ["Climate Protection Act", "NDC", "Germany", "65%", "emissions"]
    },
    {
        "id": "law_vs_pledge_02",
        "category": "law_vs_pledge",
        "question": "How does the USA's domestic legislation (Inflation Reduction Act) compare to its NDC commitments?",
        "geography_iso": "USA",
        "expected_namespace": "all",
        "ground_truth_keywords": ["Inflation Reduction Act", "NDC", "50-52%", "reduction", "emissions"]
    },
    {
        "id": "law_vs_pledge_03",
        "category": "law_vs_pledge",
        "question": "Is the UK's carbon budget under the Climate Change Act aligned with its updated NDC commitments?",
        "geography_iso": "GBR",
        "expected_namespace": "all",
        "ground_truth_keywords": ["Climate Change Act", "NDC", "carbon budget", "68%"]
    },
    {
        "id": "law_vs_pledge_04",
        "category": "law_vs_pledge",
        "question": "Is France's statutory target in its Energy-Climate Law aligned with its contribution under the EU NDC?",
        "geography_iso": "FRA",
        "expected_namespace": "all",
        "ground_truth_keywords": ["Energy-Climate Law", "EU NDC", "France", "55%", "neutrality"]
    },
    {
        "id": "law_vs_pledge_05",
        "category": "law_vs_pledge",
        "question": "How does Australia's Climate Change Act 2022 targets align with its updated 2022 NDC commitment?",
        "geography_iso": "AUS",
        "expected_namespace": "all",
        "ground_truth_keywords": ["Climate Change Act", "NDC", "Australia", "43%"]
    },
    {
        "id": "law_vs_pledge_06",
        "category": "law_vs_pledge",
        "question": "Does Canada's Net-Zero Emissions Accountability Act match its updated NDC target of 40-45% reduction?",
        "geography_iso": "CAN",
        "expected_namespace": "all",
        "ground_truth_keywords": ["Net-Zero", "Accountability Act", "NDC", "40-45%"]
    },
    {
        "id": "law_vs_pledge_07",
        "category": "law_vs_pledge",
        "question": "How do Brazil's National Policy on Climate Change (PNMC) targets compare to its international NDC pledge?",
        "geography_iso": "BRA",
        "expected_namespace": "all",
        "ground_truth_keywords": ["PNMC", "NDC", "Brazil", "reduction", "pledge"]
    },
    {
        "id": "law_vs_pledge_08",
        "category": "law_vs_pledge",
        "question": "Does Japan's Act on Promotion of Global Warming Countermeasures align with its 46% emissions reduction NDC?",
        "geography_iso": "JPN",
        "expected_namespace": "all",
        "ground_truth_keywords": ["Global Warming Countermeasures", "NDC", "Japan", "46%"]
    },
    {
        "id": "law_vs_pledge_09",
        "category": "law_vs_pledge",
        "question": "How does South Korea's Framework Act on Carbon Neutrality and Green Growth match its 2030 NDC target of 40%?",
        "geography_iso": "KOR",
        "expected_namespace": "all",
        "ground_truth_keywords": ["Framework Act", "NDC", "South Korea", "40%"]
    },
    {
        "id": "law_vs_pledge_10",
        "category": "law_vs_pledge",
        "question": "Does Mexico's General Climate Change Law align with its updated NDC commitments of 35% unconditional reduction?",
        "geography_iso": "MEX",
        "expected_namespace": "all",
        "ground_truth_keywords": ["General Climate Change Law", "NDC", "Mexico", "35%"]
    },
    {
        "id": "law_vs_pledge_11",
        "category": "law_vs_pledge",
        "question": "Compare the domestic policies under China's 1+N policy framework with its carbon peaking NDC goal before 2030.",
        "geography_iso": "CHN",
        "expected_namespace": "all",
        "ground_truth_keywords": ["1+N", "NDC", "China", "peaking", "2030"]
    },
    {
        "id": "law_vs_pledge_12",
        "category": "law_vs_pledge",
        "question": "Is the United Kingdom's Net Zero Strategy legally backed by the Climate Change Act and consistent with its 68% NDC target?",
        "geography_iso": "GBR",
        "expected_namespace": "all",
        "ground_truth_keywords": ["Net Zero Strategy", "Climate Change Act", "68%", "NDC"]
    },
    {
        "id": "law_vs_pledge_13",
        "category": "law_vs_pledge",
        "question": "Does South Africa's draft Climate Change Bill align with the emissions trajectory in its updated NDC?",
        "geography_iso": "ZAF",
        "expected_namespace": "all",
        "ground_truth_keywords": ["Climate Change Bill", "NDC", "South Africa", "emissions range"]
    },
    {
        "id": "law_vs_pledge_14",
        "category": "law_vs_pledge",
        "question": "Does Italy's National Energy and Climate Plan (PNIEC) meet the emission targets defined in the EU NDC?",
        "geography_iso": "ITA",
        "expected_namespace": "all",
        "ground_truth_keywords": ["PNIEC", "EU NDC", "Italy", "energy", "emissions"]
    },
    {
        "id": "law_vs_pledge_15",
        "category": "law_vs_pledge",
        "question": "How does the USA's Clean Air Act regulatory scope compare to the targets pledged in its 2021 NDC?",
        "geography_iso": "USA",
        "expected_namespace": "all",
        "ground_truth_keywords": ["Clean Air Act", "NDC", "USA", "emissions", "regulation"]
    },
    {
        "id": "law_vs_pledge_16",
        "category": "law_vs_pledge",
        "question": "Compare Turkey's Climate Law draft to its updated NDC pledge of a 41% reduction by 2030.",
        "geography_iso": "TUR",
        "expected_namespace": "all",
        "ground_truth_keywords": ["Climate Law", "NDC", "Turkey", "41%"]
    },
    {
        "id": "law_vs_pledge_17",
        "category": "law_vs_pledge",
        "question": "Does Argentina's Law on Minimum Standards for Global Climate Change Adaptation and Mitigation align with its NDC absolute cap?",
        "geography_iso": "ARG",
        "expected_namespace": "all",
        "ground_truth_keywords": ["Minimum Standards", "NDC", "Argentina", "adaptation", "mitigation"]
    },
    {
        "id": "law_vs_pledge_18",
        "category": "law_vs_pledge",
        "question": "Does Indonesia's Law on Environmental Protection and Management align with its updated NDC targets?",
        "geography_iso": "IDN",
        "expected_namespace": "all",
        "ground_truth_keywords": ["Environmental Protection", "NDC", "Indonesia", "targets"]
    },
    {
        "id": "law_vs_pledge_19",
        "category": "law_vs_pledge",
        "question": "How does Saudi Arabia's national circular carbon economy program align with its NDC pledge to reduce 278 MtCO2e annually?",
        "geography_iso": "SAU",
        "expected_namespace": "all",
        "ground_truth_keywords": ["circular carbon economy", "NDC", "Saudi Arabia", "278 million tons"]
    },
    {
        "id": "law_vs_pledge_20",
        "category": "law_vs_pledge",
        "question": "How does Russia's Federal Law on Limiting Greenhouse Gas Emissions compare with its NDC target of 70% of 1990 levels?",
        "geography_iso": "RUS",
        "expected_namespace": "all",
        "ground_truth_keywords": ["Limiting Greenhouse Gas", "NDC", "Russia", "70%", "1990"]
    }
]

def run_local_evaluation(pipeline: Any, chain: Any) -> Dict[str, Any]:
    """
    Evaluates the retrieval pipeline locally on the 60-query eval set.
    Calculates context recall (based on country ISO & namespace match)
    and confidence scores.
    """
    print(f"Starting local evaluation on {len(EVAL_QUERIES)} queries...")
    
    results = []
    refused_count = 0
    correct_retrieval_count = 0
    total_latency = 0.0
    
    for item in EVAL_QUERIES:
        start_time = time.time()
        
        # Run retrieval pipeline (routing, retrieval, RRF, reranking, confidence gate)
        res = pipeline.query(item["question"])
        
        latency = time.time() - start_time
        total_latency += latency
        
        refused = res["refused"]
        confidence = res["confidence_score"]
        ns_searched = res["namespace_searched"]
        retrieved_chunks = res["retrieved_chunks"]
        
        if refused:
            refused_count += 1
            
        # Context Recall Metric: Did retrieval find a chunk from the target country?
        target_iso = item["geography_iso"]
        found_target = False
        
        # Check retrieved chunks metadata (top 5)
        for chunk in retrieved_chunks:
            chunk_iso = chunk.get("metadata", {}).get("geography_iso", "")
            if chunk_iso == target_iso or (target_iso == "EU" and chunk_iso in ["EU", "EUR", "EUE"]):
                found_target = True
                break
                
        # If the query is refused, we still check candidates from the retrieve step
        # to assess whether the retrieval actually found the document
        if not found_target:
            # Check top-5 candidates before refusal
            candidates = pipeline.retrieve(item["question"], target_namespace=ns_searched, top_n=5)
            for chunk in candidates:
                chunk_iso = chunk.get("metadata", {}).get("geography_iso", "")
                if chunk_iso == target_iso or (target_iso == "EU" and chunk_iso in ["EU", "EUR", "EUE"]):
                    found_target = True
                    break
        
        if found_target:
            correct_retrieval_count += 1
            
        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "refused": refused,
            "confidence_score": confidence,
            "context_recalled": found_target,
            "latency": latency
        })
        
    num_queries = len(EVAL_QUERIES)
    avg_confidence = sum(r["confidence_score"] for r in results) / num_queries
    avg_latency = total_latency / num_queries
    recall_rate = correct_retrieval_count / num_queries
    refusal_rate = refused_count / num_queries
    
    summary = {
        "total_queries": num_queries,
        "average_confidence": avg_confidence,
        "context_recall": recall_rate,
        "refusal_rate": refusal_rate,
        "average_latency_seconds": avg_latency,
        "results": results
    }
    
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Total Queries: {num_queries}")
    print(f"Context Recall (Country ISO Match): {recall_rate:.2%}")
    print(f"Refusal Rate: {refusal_rate:.2%}")
    print(f"Average Confidence Score: {avg_confidence:.4f}")
    print(f"Average Latency: {avg_latency:.2f}s")
    print("==========================\n")
    
    # Save evaluation results to disk
    os.makedirs("data", exist_ok=True)
    with open("data/eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary

if __name__ == "__main__":
    # Save the evaluation set to data/eval_set.json
    os.makedirs("data", exist_ok=True)
    with open("data/eval_set.json", "w") as f:
        json.dump(EVAL_QUERIES, f, indent=2)
    print("Evaluation set saved to data/eval_set.json")
