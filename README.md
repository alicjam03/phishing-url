# 🛡️ Phishing URL Detection using Machine Learning

This project explores the application of machine learning techniques to detect and understand phishing URLs — a growing threat in the cybersecurity space, particularly post-COVID-19. The work focuses on three core challenges: **classification**, **clustering**, and **anomaly detection** using both supervised and unsupervised approaches.

## 📌 Project Goals

- **Classify** phishing vs legitimate URLs using supervised learning
- **Cluster** URLs to explore hidden patterns using unsupervised learning
- **Detect anomalies** that may indicate malicious behavior using hybrid techniques

## 🧠 Techniques & Tools

| Task | Algorithms Used | Best Performer |
|------|------------------|----------------|
| **Classification** | Decision Tree, Gaussian Naïve Bayes, Ensemble methods (XGBoost) | ✅ XGBoost |
| **Clustering** | K-Means++, DEC (Deep Embedded Clustering), Hierarchical, DBSCAN | ✅ K-Means++ |
| **Anomaly Detection** | Variational Autoencoders (VAE), Isolation Forest, with Word2Vec embedding | ✅ VAE + Word2Vec |

## 🔍 Key Insights

- **XGBoost** achieved perfect accuracy while maintaining fast training time.
- **K-Means++** offered the best trade-off between performance and interpretability in clustering.
- **Word2Vec embedding** significantly improved performance in detecting anomalies with VAE.
- Preprocessing involved **balancing the dataset** and **feature selection** to reduce memory and improve computational efficiency.

## 📊 Dataset

A large phishing URL dataset was used, requiring careful preprocessing due to:
- Class imbalance
- High dimensionality
- Memory limitations

Techniques applied:
- Feature selection
- Sampling to balance classes
- Text vectorization (e.g., Word2Vec)

## 💻 Technologies Used

- Python
- scikit-learn
- XGBoost
- Keras / TensorFlow
- Word2Vec (gensim)
- Matplotlib / Seaborn for visualizations
- Jupyter Notebook


## 📈 Results Overview

- **Classification:**  Accuracy > 95% (XGBoost)
- **Clustering:**      Silhouette scores high, but ARI lower — best balance with K-Means++
- **Anomaly Detection:** Word2Vec improved results significantly; VAE performed best overall

## 🚀 Future Improvements

- Combine models in an ensemble across tasks
- Explore deep NLP-based feature extraction (e.g., BERT)
- Deploy as a real-time detection tool


## 🤝 Acknowledgements

This project was completed as part of the MSc Artificial Intelligence course at Birmingham City University.

