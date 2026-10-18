# Credits & Academic References

FairScreen acknowledges and credits the following research and open-source foundations:

1. **AI Fairness 360 (AIF360)**
   - *IBM Research*: Bellamy et al. (2018). "AI Fairness 360: An extensible toolkit for detecting, understanding, and mitigating unwanted algorithmic bias."
   - Specifically utilized for the sample reweighing mitigation formulation:
     $$W(S=s, Y=y) = \frac{P(S=s) \times P(Y=y)}{P(S=s, Y=y)}$$

2. **Bias in Bios Benchmark Dataset**
   - De-Arteaga, M., Romanov, A., Wallach, H., Chayes, J., Borgs, C., Chouldechova, A., Geyik, S., Kenthapadi, K., & Kalai, A. (2019). "Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes Setting." *ACM Conference on Fairness, Accountability, and Transparency (FAT\* 2019)*. HuggingFace: `LabHC/bias_in_bios`.

3. **PopResume Causal Fairness Benchmark**
   - Yu, S., et al. (2026). "PopResume: Population-Representative Resume Generation and Causal Fairness Auditing in Automated Recruitment." *EMNLP Oral Presentation*. HuggingFace: `sumin-yu/PopResume`.

4. **Core Open-Source Libraries**
   - **Streamlit**: Dashboard framework for ethical machine learning interfaces.
   - **scikit-learn**: Classical predictive modeling (Logistic Regression, Gradient Boosting, TF-IDF).
   - **pypdf & pdfplumber**: In-memory PDF parsing and structural text extraction.
