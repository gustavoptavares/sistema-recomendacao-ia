import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from backend.agents.graph import graph

st.set_page_config(page_title="AI Book Recommender", layout="wide")

st.title("📚 Sistema de Recomendação de Livros com IA")
st.markdown("### Powered by Google Books, LangGraph & RAG")

with st.sidebar:
    st.header("Opções")
    st.info("Este sistema utiliza agentes inteligentes para analisar sua intenção, buscar livros e classificar os melhores resultados.")

query = st.text_input("O que você gostaria de ler hoje?", placeholder="Ex: Livros sobre história da computação...")

if st.button("Buscar Recomendações") and query:
    with st.spinner("Os agentes estão trabalhando... (Analisando intenção -> RAG -> Rerank -> Explicação)"):
        try:
            result = graph.invoke({"query": query})
            recs = result.get("recommendations", [])
            exps = result.get("explanations", [])
            
            if not recs:
                st.warning("Nenhum livro encontrado de relevante.")
            else:
                st.success(f"Encontramos {len(recs)} recomendações personalizadas!")
                
                for i, book in enumerate(recs):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if book.get('thumbnail'):
                            st.image(book['thumbnail'], width=100)
                        else:
                            st.write("🖼️ Sem capa")
                    with col2:
                        st.subheader(f"{i+1}. {book.get('title', 'Sem título')}")
                        authors = book.get('authors', ['Desconhecido'])
                        if isinstance(authors, list):
                            authors = ', '.join(authors)
                        st.write(f"**Autor(es):** {authors}")
                        st.write(f"**Ano:** {book.get('published_year', 'N/A')}")
                        if i < len(exps):
                            st.info(f"💡 **Por que:** {exps[i]}")
                        with st.expander("Ver descrição completa"):
                            st.write(book.get('description', 'Sem descrição disponível.'))
                    st.divider()
        except Exception as e:
            st.error(f"Erro: {e}")

st.markdown("---")
st.caption("POC Version 1.0 | Google Books | LangGraph | LangSmith")