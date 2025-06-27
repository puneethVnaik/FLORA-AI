from fastapi import FastAPI, Form, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from pydantic import BaseModel
import json

app = FastAPI()

# ------------------ CORS & Static ------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images/", StaticFiles(directory="images"), name="images")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# ------------------ Medicines DB Setup ------------------

with open("grouped_medicines.json", "r") as file:
    med_data = json.load(file)

med_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
med_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-MiniLM-L6-v2")
med_client = chromadb.PersistentClient(path="./medicines_db")

if "medicines" in [c.name for c in med_client.list_collections()]:
    med_collection = med_client.get_collection(name="medicines", embedding_function=med_embed_fn)
else:
    med_collection = med_client.create_collection(name="medicines", embedding_function=med_embed_fn)

if med_collection.count() == 0:
    entries = []
    for issue, meds in med_data.items():
        for idx, med in enumerate(meds):
            entry_id = f"{issue}_{idx}"
            text = f"{issue}: {med['Name of Medicine']}, Dose: {med['Dose and Mode of Administration']}, Indication: {med['Indication']}"
            entries.append({
                "id": entry_id,
                "document": text,
                "metadata": {
                    "health_issue": issue,
                    "medicine": med["Name of Medicine"],
                    "dose": med["Dose and Mode of Administration"],
                    "indication": med["Indication"]
                }
            })
    med_collection.add(
        documents=[e["document"] for e in entries],
        metadatas=[e["metadata"] for e in entries],
        ids=[e["id"] for e in entries]
    )

# ------------------ Plant DB Setup ------------------

with open("plants1.json", "r", encoding="utf-8") as f:
    plant_data = json.load(f)

plant_model = SentenceTransformer("all-MiniLM-L6-v2")
plant_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
plant_client = chromadb.PersistentClient(path="./plants_db")

if "plants" in [c.name for c in plant_client.list_collections()]:
    plant_collection = plant_client.get_collection(name="plants", embedding_function=plant_embed_fn)
else:
    plant_collection = plant_client.create_collection(name="plants", embedding_function=plant_embed_fn)

if plant_collection.count() == 0:
    plant_entries = []
    for idx, plant in enumerate(plant_data):
        doc = f"{plant.get('Plant Name', '')}, {plant.get('Scientific Name', '')}, {plant.get('Uses', '')}, {plant.get('Healing Properties', '')}"
        plant_entries.append({
            "id": f"plant_{idx}",
            "document": doc,
            "metadata": plant
        })
    plant_collection.add(
        documents=[e["document"] for e in plant_entries],
        metadatas=[e["metadata"] for e in plant_entries],
        ids=[e["id"] for e in plant_entries]
    )

# ------------------ Routes ------------------

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/medicine_search")
async def medicine_search(query: str):
    result = med_collection.query(query_texts=[query], n_results=5)
    if not result["metadatas"]:
        return JSONResponse(content={"message": "No matching health issue found."}, status_code=404)

    top_issue = result["metadatas"][0][0]["health_issue"]
    all_meds = med_collection.get()
    related = [m for m in all_meds["metadatas"] if m["health_issue"] == top_issue]

    return {
        "health_issue": top_issue,
        "results": related
    }


class QueryRequest(BaseModel):
    query: str


BASE_IMAGE_URL = "https://ayur-b547.onrender.com/images/"


def search_plants(query: str, top_k: int = 1):
    embedding = plant_model.encode([query]).tolist()
    results = plant_collection.query(query_embeddings=embedding, n_results=top_k)

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    output = []
    for doc, meta in zip(docs, metas):
        image_file = meta.get("Image", "")
        image_url = BASE_IMAGE_URL + image_file.replace("images/", "") if image_file else None
        output.append({
            "Plant Name": meta.get("Plant Name", ""),
            "Scientific Name": meta.get("Scientific Name", ""),
            "Healing Properties": meta.get("Healing Properties", ""),
            "Uses": meta.get("Uses", ""),
            "Description": meta.get("Description", ""),
            "Preparation Method": meta.get("Preparation Method", ""),
            "Side Effects": meta.get("Side Effects", ""),
            "Geographic Availability": meta.get("Geographic Availability", ""),
            "Image": image_url,
            "Image Missing": not bool(image_file)
        })
    return output


@app.post("/search/")
async def search_post(req: QueryRequest):
    return {"results": search_plants(req.query)}


@app.get("/search/")
async def search_get(query: str = Query(...)):
    return {"results": search_plants(query)}


@app.get("/ask")
async def ask(query: str = Query(...)):
    return {"results": search_plants(query, top_k=3)}


@app.get("/plant_names")
def plant_names():
    result = plant_collection.query(query_texts=["*"], n_results=1000)
    if "metadatas" not in result:
        return []
    names = list({meta["Plant Name"] for meta in result["metadatas"] if "Plant Name" in meta})
    return names
