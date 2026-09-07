"""
Prompt construction and system template module for the HealthMate application.

This module defines the core system prompt guidelines and utility functions 
to structure context-augmented prompts for RAG-based responses.
"""

SYSTEM_PROMPT = """Kamu adalah HealthMate, asisten AI yang bantu jawab pertanyaan seputar kesehatan
umum dengan gaya santai dan gampang dimengerti (seperti ngobrol sama teman yang paham medis dasar),
BUKAN dengan bahasa kaku ala buku teks kedokteran.

ATURAN PENTING:
1. Jawab HANYA berdasarkan konteks/dokumen yang diberikan di bawah. Jangan mengarang informasi medis
   di luar konteks tersebut.
2. Jika konteks yang diberikan tidak relevan atau tidak cukup untuk menjawab, katakan dengan jujur
   bahwa kamu tidak punya informasi yang cukup akurat untuk itu -- jangan menebak-nebak soal kesehatan.
3. Kamu BUKAN pengganti dokter. Untuk gejala serius, mendadak, atau darurat, selalu sarankan
   pengguna untuk segera konsultasi ke tenaga medis profesional atau ke IGD terdekat.
4. Jangan memberikan diagnosis pasti maupun resep dosis obat spesifik ke pengguna individu.
   Sampaikan informasi umum saja, lalu arahkan ke profesional medis untuk keputusan personal.
5. Gaya bahasa: santai, hangat, pakai bahasa Indonesia sehari-hari, boleh emoji secukupnya,
   tapi tetap sopan dan informasinya tetap akurat sesuai konteks.
6. Di akhir jawaban yang membahas kondisi/gejala, selipkan singkat pengingat untuk konsultasi
   profesional jika relevan (tidak perlu di setiap pesan kalau obrolannya ringan/basa-basi).
"""


def build_user_turn(query: str, retrieved_docs: list) -> str:
    """
    Constructs a formatted user prompt combining retrieved medical documents 
    and the user's query for the LLM context injection.

    Args:
        query (str): The question or query provided by the user.
        retrieved_docs (list): List of document dictionaries retrieved from the vector store.

    Returns:
        str: A formatted string block combining context and the query.
    """

    # Fallback if no relevant documents were returned from vector store
    if not retrieved_docs:
        context_block = "(Tidak ada dokumen relevan ditemukan di knowledge base.)"
    else:
        parts = []
        for i, doc in enumerate(retrieved_docs, start=1):
            # Format each retrieved document metadata and QA pair into a structured block
            parts.append(
                f"[Dokumen {i}] Topik: {doc.get('focus') or '-'} | "
                f"Jenis: {doc.get('qtype') or '-'}\n"
                f"Q: {doc.get('question')}\n"
                f"A: {doc.get('answer')}"
            )
        context_block = "\n\n".join(parts)

    # Combine knowledge base context block with user query into final execution prompt
    return (
        f"KONTEKS DARI KNOWLEDGE BASE (sumber: MedQuAD/NIH):\n{context_block}\n\n"
        f"PERTANYAAN USER:\n{query}\n\n"
        f"Jawab dengan gaya santai sesuai instruksi system, berdasarkan konteks di atas."
    )
