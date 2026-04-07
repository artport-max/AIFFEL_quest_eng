#!/usr/bin/env python3
"""
ArtGlaze Cloud — Glaze Knowledge RAG 검색 모듈
ChromaDB의 Digitalfire KB를 검색하여 유약 분석 컨텍스트 반환
"""

import json
import os
import chromadb
from chromadb.utils import embedding_functions

# ── 경로 설정 (프로젝트 상대 경로) ──────────────────────
# 이 파일 위치: scripts/glaze_rag.py
# 프로젝트 루트: 한 단계 위
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR      = os.path.join(BASE_DIR, "chromadb")
OXIDE_REF_PATH  = os.path.join(BASE_DIR, "data", "digitalfire_db", "oxide_quick_ref.json")
DEFECT_PATH     = os.path.join(BASE_DIR, "data", "digitalfire_db", "defect_diagnosis.json")

COLLECTION_NAME = "artglaze_digitalfire"
EMBED_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"


class GlazeRAG:
    def __init__(self):
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn
        )

        with open(OXIDE_REF_PATH, encoding="utf-8") as f:
            self.oxide_ref = json.load(f)
        with open(DEFECT_PATH, encoding="utf-8") as f:
            self.defect_ref = json.load(f)

        print(f"GlazeRAG 초기화 완료 — 컬렉션: {COLLECTION_NAME}")

    def search(self, query: str, n: int = 5, category: str = None) -> list:
        where = {"source": "digitalfire"}
        if category:
            where["category"] = category
        results = self.collection.query(
            query_texts=[query],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            output.append({
                "term":      meta.get("term", ""),
                "category":  meta.get("category", ""),
                "relevance": round(1 - dist, 4),
                "text":      doc,
                "tags":      meta.get("tags", "").split(","),
                "url":       meta.get("url", ""),
            })
        return output

    def get_oxide(self, oxide: str) -> dict:
        return self.oxide_ref.get(oxide, None)

    def diagnose_defect(self, defect: str) -> dict:
        return self.defect_ref.get(defect, None)

    def build_analysis_context(self, recipe: dict, atmosphere: str = "oxidation") -> str:
        context_parts = []
        for oxide, value in recipe.items():
            ref = self.get_oxide(oxide)
            if ref:
                context_parts.append(
                    f"[{oxide} = {value}]\n"
                    f"정의: {ref['definition']}\n"
                    f"효과: {ref['key_effects']}\n"
                )
        atm_results = self.search(
            f"{atmosphere} firing glaze color", n=2, category="firing_process"
        )
        for r in atm_results:
            context_parts.append(f"[소성:{r['term']}]\n{r['text'][:200]}\n")

        chem_results = self.search("UMF stability", n=2, category="chemistry_method")
        for r in chem_results:
            context_parts.append(f"[화학:{r['term']}]\n{r['text'][:200]}\n")

        return "\n---\n".join(context_parts)

    def risk_score(self, umf: dict) -> dict:
        sio2  = umf.get("SiO2",  0)
        al2o3 = umf.get("Al2O3", 0)
        na2o  = umf.get("Na2O",  0)
        k2o   = umf.get("K2O",   0)
        si_al = sio2 / al2o3 if al2o3 > 0 else 99
        alkali = na2o + k2o

        if alkali > 0.4 or sio2 < 2.5:
            crazing = "high"
        elif alkali > 0.25 or sio2 < 3.0:
            crazing = "medium"
        else:
            crazing = "low"

        if si_al > 8:
            surface = "glossy"
        elif si_al > 5:
            surface = "satin"
        else:
            surface = "matte"

        return {
            "crazing_risk": crazing,
            "surface_type": surface,
            "si_al_ratio":  round(si_al, 2),
            "alkali_total": round(alkali, 3),
        }
