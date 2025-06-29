# 🌿 FloraAI: Real-Time Medicinal Plant Identification Using AI and Mobile Vision

FloraAI is an AI-powered mobile application designed to help users identify medicinal plants in real time and explore rich information about their uses, scientific properties, preparation methods, and more.  
Built using advanced deep learning, semantic search, and mobile vision, FloraAI bridges the gap between modern technology and traditional medicinal knowledge.

---

## 🚀 Project Overview

- **Real-Time Plant Identification**:  
  Capture a live image of a medicinal plant using your mobile camera to get instant predictions.

- **Smart Search**:  
  Search for plant names or health conditions to discover detailed information, including:
  - Scientific name
  - Medicinal properties
  - Preparation methods
  - Side effects
  - Geographic availability
  - Images and Google Maps location

- **Modern Tech Stack**:  
  Combining CNN (ResNet50), RAG architecture, semantic embeddings, and Android (Kotlin) development.

---

## 🧠 Technical Highlights

### 🌱 Medicinal Plant Identification
- Fine-tuned **ResNet50** convolutional neural network.
- Added **5 custom layers** for improved classification.
- Trained on a dataset of **50 medicinal plant species** (~30,000 images total).
- Integrated into a mobile app for **real-time predictions**.

### 🔍 AI-Powered Search (RAG)
- Data prepared from **trusted botanical survey websites** into a structured Excel dataset.
- Converted into documents and chunked using **RecursiveCharacterTextSplitter**.
- Embedded using **HuggingFace all-MiniLM-L6-v2**.
- Stored in **ChromaDB** for fast semantic retrieval.
- Supports natural language queries like:
  > “Which plant helps with cough?”  
  > “Tell me about Neem”

- Results formatted in markdown with clear headers, images, and Google Maps links.

### 📱 Mobile App
- Developed in **Android Studio** with **Kotlin**.
- Features:
  - Camera integration for live identification.
  - Search bar for health conditions or plant names.
- Backend powered by CNN model and RAG system.

---

## 🛠 Tech Stack

| Module                         | Technology                                     |
|--------------------------------|-----------------------------------------------|
| Model                          | CNN (ResNet50 + custom layers)                |
| Semantic Search                | LangChain, ChromaDB, HuggingFace Embeddings   |
| Backend Prototyping            | Streamlit                                     |
| Mobile Development             | Android Studio (Kotlin)                       |
| Dataset Management             | Excel / CSV files                             |

---

## 📦 Project Structure

FloraAI/
- ├── app/ # Android (Kotlin) mobile app source
- ├── model/ # CNN model scripts and training notebooks
- ├── rag/ # RAG pipeline scripts (embedding, storage, retrieval)
- ├── data/ # plants.xlsx and related datasets
- ├── streamlit_app/ # Prototype interface
- ├── README.md # Project documentation
- └── requirements.txt # Python dependencies

## ✨ Key Features

✅ Identify 50+ medicinal plants instantly using your camera  
✅ Search by plant name or health condition  
✅ Get detailed medicinal uses, side effects, and preparation tips  
✅ Google Maps integration for location references  
✅ Modern and user-friendly mobile interface

---

## 📊 Results

- Achieved high accuracy on validation dataset.
- Real-time prediction latency optimized for mobile devices.
- Rich, semantically relevant search responses powered by AI.

---

## 📝 References

- Data sourced from **National Botanical Survey websites** and trusted medicinal plant databases.
- Powered by **ResNet50**, **LangChain**, **ChromaDB**, and **HuggingFace**.

---

## 📍 Future Enhancements

- Multi-language support
- Voice-enabled search and Q&A
- Expanded plant database
- AR-based plant information overlay

---

## 🤝 Contributing

We welcome contributions!  
Please open issues, submit pull requests, or share feedback to help us grow FloraAI.

---

## 📜 License

This project is licensed under the **MIT License**.

---

> 🌿 **FloraAI** — Bridging nature and AI to empower everyone to learn about medicinal plants, anywhere, anytime.
