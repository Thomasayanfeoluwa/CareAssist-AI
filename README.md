# CareAssist-AI

# End-to-end-Medical-Chatbot-Generative-AI


# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/Thomasayanfeoluwa/CareAssist-AI.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n Carebot python=3.10 -y
```

```bash
conda activate Careibot
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
GROQ_API_KEY="======================================"
PINECONE_API_KEY="**************************************"
GOOGLE_CSE_ID="==================="
GOOGLE_CSE_API_KEY = "**************************************"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py for Flask
streamlit run dashboard.py for Streamlit
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask/Streamlit/Chainlit
- Google Client
- Groq
- Pinecone







<img width="1365" height="689" alt="Screenshot (121)" src="https://github.com/user-attachments/assets/60388cbe-9c87-4149-84ca-4fab2accac3c" />




https://github.com/user-attachments/assets/a6bd51b3-29c3-48a5-8054-b06e5f1b838f



