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
- **Streamlit** (for UI)
- **Colab/Jupyter**
- **xAI Libraries**

---

## ✨ Screenshots

> Add your screenshots here!  
> ![Screenshot Placeholder](https://via.placeholder.com/800x400.png?text=Visualization+Dashboard+Screenshot)

---

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
