# RAG-With-RAPTOR
## Prerequisites
Docker Desktop

GitBash

## Steps to Run the Code

### 1. Clone the Repository
1.Open GitBash.

2.Create a new folder and copy the path.

3.Navigate to the folder in gitbash using:
cd "your-folder-path"

4.Clone the repository:
git clone https://github.com/AdithMurari/RAG-With-RAPTOR.git

### 2. Download and Setup Docker Compose
1.Download the YAML file from the link below:
https://github.com/milvus-io/milvus/releases/download/v2.3.19/milvus-standalone-docker-compose.yml

2.Store the file in the new folder you created and rename it to docker-compose.yml

### 3. Open the Project in an IDE
1.Open the folder in any IDE (preferably VS Code).

2.Open the GitBash terminal.

3.Run the command:
docker-compose up -d

### 4. Create and Activate a Virtual Environment
1.Open a new terminal and create virtual environment:
python -m venv myenv

2.Activate the virtual environment:
\myenv\Scripts\activate (for Windows)

### 5. Install Dependencies and Setup Environment Variables
1.Install all the dependencies using the command:
pip install -r requirements.txt

2.Create a .env file to store the environment variables like OpenAI and Langsmith API keys.

### 6.Ingest Data
Open the ingest.ipynb file.

Select the kernel and run the file. This will:

-->Ingest the data.

-->Convert the data into chunks.

-->Perform RAPTOR indexing.

-->Store the indexed data in the Milvus vector database

### 7.Launch the Chatbot Interface
Open your terminal and run the command:
streamlit run APP.py

### Viola
You now have a chatbot for the stock market ready to use.

