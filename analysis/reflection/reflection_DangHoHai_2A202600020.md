# Personal Reflection - Lab Day 14: AI Evaluation & Benchmarking

**Full Name:** Dang Ho Hai  
**Student ID:** 2A202600020  
**Group:** D5-2  

---

## 1. Engineering Contribution

In this project, I contributed to developing core features and enhancing the system's evaluation capabilities:

* **Data Ingestion & Preparation (`data/` module):**
    * Implemented necessary scripts and configurations to ingest data into the system, ensuring data is properly formatted for the benchmarking process.
    * Contributed to creating the **Golden Dataset**, specifically test cases with **Ground Truth IDs**, to support the calculation of retrieval metrics such as Hit Rate and MRR.
* **Semantic Search Integration (`agent/` or `engine/` module):**
    * Researched and integrated **Semantic Search** functionality into the system. This involved updating the Agent's retrieval mechanism and improving how the system searches and matches relevant documents.
    * Ensured the semantic search feature operates efficiently, enabling the Agent to understand and respond to more complex queries, thereby improving response quality.

---

## 2. Technical Depth

During the implementation of the above contributions, I researched and applied several key technical concepts:

* **Hit Rate and MRR (Mean Reciprocal Rank):** I gained a deep understanding of the importance of these metrics in evaluating the quality of the Retrieval stage. **Hit Rate** measures how often the correct document is found, while **MRR** evaluates not just the presence but also the rank of the correct document in the results list—critical for mitigating the *"Lost in the Middle"* phenomenon in LLMs.
* **Golden Dataset and Ground Truth IDs:** I mastered the process of building a Golden Dataset with Ground Truth IDs, which serves as the foundation for accurately measuring retrieval performance and determining if the system retrieves the appropriate information "chunks."
* **Performance vs. Cost Trade-offs:** When integrating more complex models or search mechanisms, I considered factors such as performance (retrieval speed) and cost (API usage, computational resources) to aim for an optimal solution.

---

## 3. Problem Solving

During the workflow, I encountered and resolved several technical challenges:

1.  **Ensuring Data Quality:** When adding new data, ensuring consistency and quality is vital to avoid biasing evaluation results. I proactively verified and cleaned the data to ensure it met the benchmark requirements.
2.  **Effective Semantic Search Integration:** Integrating a complex feature like Semantic Search requires a deep understanding of how models operate and interact with Vector Databases. I experimented with various approaches to ensure the feature significantly improved system capabilities without causing unintended side effects.
3.  **Handling Exceptions:** While preparing data and integrating search, I encountered non-standard data formats and unexpected search results. I developed error-handling and exception mechanisms to ensure the evaluation pipeline remains stable and provides reliable results.

---
*I hereby declare that the information provided above is true and accurately reflects my contributions to the project.*
