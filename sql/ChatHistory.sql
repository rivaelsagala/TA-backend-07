-- ==========================================
-- INIT SCHEMA PERDES-AI DATABASE
-- ==========================================

-- 1. Drop tabel jika sudah ada agar script aman dijalankan berulang (Clean Slate)
DROP TABLE IF EXISTS chunks_perdes CASCADE;
DROP TABLE IF EXISTS chat_history CASCADE;
DROP TABLE IF EXISTS chat_sessions CASCADE;
DROP TABLE IF EXISTS users CASCADE;


-- ==========================================
-- CREATE TABLES
-- ==========================================

-- 2. Table users
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL, -- Dihapus constraint UNIQUE-nya
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Table chat_sessions
CREATE TABLE chat_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_name VARCHAR(255),
    evaluate BOOLEAN DEFAULT FALSE, -- Default false agar terdefinisi di awal
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Table chat_history
CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    user_query TEXT NOT NULL,
    llm_response TEXT NOT NULL,
    metadata JSONB,
    
    -- Status Evaluasi
    is_evaluated BOOLEAN DEFAULT FALSE, 
    
    -- RAG Evaluation Metrics (Digabung langsung ke tabel)
    faithfulness FLOAT DEFAULT NULL,
    answer_relevance FLOAT DEFAULT NULL,
    context_precision FLOAT DEFAULT NULL,
    context_recall FLOAT DEFAULT NULL,
    noise_sensitivity FLOAT DEFAULT NULL,
    similarity_score FLOAT DEFAULT NULL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Table chunks_perdes (Penyimpanan Dokumen Regulasi)
CREATE TABLE chunks_perdes (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(255),
    content TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ==========================================
-- CREATE INDEXES
-- ==========================================
-- Indeks dibuat di akhir agar proses insert awal (jika ada) lebih cepat

CREATE INDEX idx_chat_history_session ON chat_history(session_id);
CREATE INDEX idx_chat_history_user ON chat_history(user_id);
CREATE INDEX idx_chat_sessions_user ON chat_sessions(user_id);
CREATE INDEX idx_chunks_perdes_file_name ON chunks_perdes(file_name);


-- CREATE TABLE users (
--     id SERIAL PRIMARY KEY,
--     username VARCHAR(255) UNIQUE NOT NULL,
--     password VARCHAR(255) UNIQUE NOT NULL,
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );

-- CREATE TABLE chat_sessions (
--     id SERIAL PRIMARY KEY,
--     user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
--     session_name VARCHAR(255),
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
--     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );

-- CREATE TABLE chat_history (
--     id SERIAL PRIMARY KEY,
--     session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
--     user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
--     user_query TEXT NOT NULL,
--     llm_response TEXT NOT NULL,
--     metadata JSONB,
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );

-- -- Create indexes
-- CREATE INDEX idx_chat_history_session ON chat_history(session_id);
-- CREATE INDEX idx_chat_history_user ON chat_history(user_id);
-- CREATE INDEX idx_chat_sessions_user ON chat_sessions(user_id);

-- CREATE TABLE chunks_perdes (
--     id SERIAL PRIMARY KEY,
--     file_name VARCHAR(255),
--     content TEXT,
--     metadata JSONB,
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );

-- CREATE INDEX idx_chunks_perdes_file_name ON chunks_perdes(file_name);




-- -- Membuat tabel khusus untuk menyimpan hasil metrik RAGAS
-- CREATE TABLE chat_evaluations (
--     id SERIAL PRIMARY KEY,
--     chat_history_id INTEGER UNIQUE NOT NULL REFERENCES chat_history(id) ON DELETE CASCADE,
--     faithfulness FLOAT,
--     answer_relevance FLOAT,
--     context_precision FLOAT,
--     context_recall FLOAT,
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );

-- -- Index untuk mempercepat pencarian data evaluasi berdasarkan history chat
-- CREATE INDEX idx_chat_evaluations_history ON chat_evaluations(chat_history_id);


-- ALTER TABLE chat_history
-- ADD COLUMN faithfulness FLOAT DEFAULT NULL,
-- ADD COLUMN answer_relevance FLOAT DEFAULT NULL,
-- ADD COLUMN context_precision FLOAT DEFAULT NULL,
-- ADD COLUMN context_recall FLOAT DEFAULT NULL,
-- ADD COLUMN noise_sensitivity FLOAT DEFAULT NULL,
-- ADD COLUMN similarity_score FLOAT DEFAULT NULL;