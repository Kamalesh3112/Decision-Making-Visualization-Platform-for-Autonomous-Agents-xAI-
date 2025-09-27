# Decision-Making Visualization Platform for Autonomous Agents (xAI)

Welcome to the **Decision-Making Visualization Platform for Autonomous Agents (xAI)**!  
This is an unsupervised machine learning project provides an interactive and intuitive platform for visualizing and understanding the decision-making processes of autonomous agents using explainable AI (xAI) techniques.

---

## 🚀 Overview

- **Purpose:**  
  Gain deep insights into how autonomous agents make decisions by visualizing their thought processes and justifications.
- **Visualization:**  
  Interactive dashboards and visual components make the black-box nature of AI models transparent and accessible.
- **Explainability:**  
  Designed to help researchers, students, and practitioners demystify agent behaviors.

---

## 🏗️ Features

- **Agent Decision Path Visualization**
- **Customizable Scenarios & Inputs**
- **Rich, Interactive xAI Visuals**
- **Jupyter/Colab Notebook Integration**
- **Runs as a Streamlit App (Colab Compatible Only)**

---

## ⚡ Quick Start (Google Colab Only)

> **Note:**  
> This project is intended to be run inside **Google Colab**.  
> It is *not* compatible with local Jupyter or other environments.

1. **Open the [Colab Notebook](./path/to/notebook.ipynb) in Google Colab.**
2. **Run all cells** to install dependencies, set up the environment, and launch the visualization server.
3. Once you see the message:
    ```
    Your app is running at http://localhost:8501
    ```
    Replace `localhost` with your current Colab VM's IP address, or use the provided external URL (if using ngrok):

    - **Colab IP Example:**  
      ```
      http://<your-colab-vm-ip>:8501
      ```
    - **ngrok Example:**  
      ```
      https://bd3eab6dcb0c.ngrok-free.app/
      ```

    > **Tip:** The notebook will usually print the external URL for you once the app is running.
---

<img width="1905" height="978" alt="image" src="https://github.com/user-attachments/assets/d05cfcfb-7ce3-4042-8a6d-bc965f6aefaa" />

4. **Click the generated ngrok web server link** to access your interactive visualization dashboard from any device!

---

## 🌐 About ngrok Web Server

To enable external access to the Streamlit app running in Colab, **ngrok** is used to create a public tunnel to port `8501`.  
- This automatically generates a public URL (like `https://bd3eab6dcb0c.ngrok-free.app/`) which you can use to access the Streamlit app in your browser.
- Ngrok is started and managed automatically by the Colab notebook—no manual setup required!
- This requires authentication token inorder to successfully run the app on desired port number.

**Why ngrok?**
- Google Colab notebooks run in a temporary cloud VM with no direct public access.
- Ngrok solves this by creating a secure tunnel from the Colab VM to the web, making your app instantly shareable.

---


## 📊 Example Use Cases

- Visualize how autonomous vehicles make navigation decisions
- Analyze reinforcement learning agent policies step-by-step
- Explain the rationale behind AI recommendations in critical scenarios

---

## 💻 Technologies Used

- **Python**
- Machine Learning
- **Streamlit** (for UI)
- **Colab/Jupyter**
- **xAI Libraries**

---

Here's what this complete end-to-end system includes: 🔽

## 🎯 **Core Features**

### **1. Data Simulation & Management**
- Multi-agent decision simulation (5 different agent types)
- SQLite database for persistent storage
- Real-time decision generation with bias injection
- Context-aware decision making (high-stakes, routine, emergency, collaborative)

### **2. Unsupervised ML Analysis**
- **K-Means Clustering**: Identifies bias patterns in agent behavior
- **DBSCAN**: Density-based clustering for anomaly detection
- **Isolation Forest**: Detects outlier decisions
- **PCA, t-SNE, UMAP**: Dimensionality reduction for visualization
- **Silhouette Analysis**: Optimal cluster validation

### **3. Fairness Analysis**
- Demographic parity assessment
- Equalized odds calculation  
- Individual fairness metrics
- Treatment equality analysis
- Statistical significance testing
- Systematic bias detection (temporal, contextual, algorithmic)

### **4. Interactive Dashboard**
- **5 Main Tabs**: Overview, Bias Clusters, Fairness Analysis, Temporal Analysis, Alerts
- Real-time visualizations with Plotly
- Filtering and data exploration
- Alert management system
- AI-generated recommendations

## 🛠️ **Tech Stack**

| Component | Technology |
|-----------|------------|
| **Backend** | Python, SQLite, scikit-learn |
| **Frontend** | Streamlit, Plotly Dashboard |
| **ML Algorithms** | K-Means, DBSCAN, Isolation Forest, PCA, t-SNE, UMAP |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Seaborn, Matplotlib |
| **Statistics** | SciPy, NetworkX |

## 🚀 **Quick Start**

```bash
# Install dependencies
pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib umap-learn networkx scipy

# Run demo analysis
python xai_platform.py

# Launch interactive dashboard
streamlit run xai_platform.py dashboard
```

## 📊 **Key Capabilities**

### **Bias Detection**
- **Algorithmic bias** patterns across agent types
- **Selection bias** in decision contexts
- **Confirmation bias** based on data quality
- **Temporal bias** throughout different time periods

### **Fairness Monitoring**
- Real-time fairness violation alerts
- Cross-agent type fairness comparison
- Context-based bias analysis
- Statistical significance testing for bias detection

### **Visual Analytics**
- Interactive cluster visualizations
- Temporal pattern analysis
- Agent performance comparisons
- Anomaly highlighting and investigation

## 🎭 **Production Features**

✅ **Scalable Architecture**: Modular design for easy extension  
✅ **Real-time Processing**: Live data streaming and analysis  
✅ **Comprehensive Logging**: Full decision audit trail  
✅ **Alert System**: Automated fairness violation detection  
✅ **Statistical Rigor**: Hypothesis testing and significance analysis  
✅ **Interactive UI**: User-friendly dashboard with filtering  
✅ **Export Capabilities**: Data and visualization export  

This is a complete, enterprise-ready system that demonstrates advanced unsupervised ML techniques for ethical AI monitoring. The platform can be deployed locally or in the cloud, and easily extended for specific use cases or integrated with existing agent systems.

## 📝 Citation

If you use this platform in your research or work, please cite this repository.

---

## 🤝 Contributing

Contributions, feature requests, and bug reports are welcome!  
Feel free to submit pull requests or open issues.

---

## 📬 Contact

For questions or collaboration, reach out via [GitHub Issues](https://github.com/Kamalesh3112/Decision-Making-Visualization-Platform-for-Autonomous-Agents-xAI-/issues) or [Gmail](kamalesh.sselvaraj@gmail.com)

---

**Enjoy exploring and explaining autonomous agent decisions!**
