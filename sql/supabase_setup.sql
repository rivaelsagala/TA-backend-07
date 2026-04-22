-- ============================================
-- Supabase pgvector Setup untuk RAG Project
-- ============================================
-- Jalankan SQL ini di Supabase SQL Editor
-- Pastikan extension pgvector sudah diaktifkan

-- 1. Enable pgvector extension
create extension if not exists vector;

-- 1.1. Drop existing table jika structure salah (HATI-HATI: ini akan hapus semua data!)
-- Uncomment baris ini jika mau reset table:
-- DROP TABLE IF EXISTS documents CASCADE;

-- 2. Buat tabel documents untuk menyimpan chunks + embeddings
-- PENTING: LangChain SupabaseVectorStore memerlukan UUID sebagai primary key
create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  metadata jsonb default '{}'::jsonb,
  embedding vector(1536),  -- OpenAI text-embedding-3-small dimension
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- 3. Buat index untuk pencarian similarity yang lebih cepat
create index if not exists documents_embedding_idx 
on documents 
using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

-- 4. Buat function untuk similarity search (match_documents)
-- PENTING: Return type harus UUID bukan bigint
create or replace function match_documents (
  query_embedding vector(1536),
  match_count int default 5,
  filter jsonb default '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- 5. Script untuk cek struktur tabel existing
SELECT 
    column_name, 
    data_type, 
    is_nullable 
FROM information_schema.columns 
WHERE table_name = 'documents' 
ORDER BY ordinal_position;

-- 6. Jika ID masih bigint, jalankan ini untuk fix:
-- ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_pkey;
-- ALTER TABLE documents ALTER COLUMN id SET DATA TYPE uuid USING gen_random_uuid();
-- ALTER TABLE documents ADD PRIMARY KEY (id);

-- 7. Verify pgvector extension
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- 5. (Optional) Buat RLS policy jika diperlukan
-- alter table documents enable row level security;
-- 
-- create policy "Enable read access for all users" on documents
--   for select using (true);
-- 
-- create policy "Enable insert for authenticated users only" on documents
--   for insert with check (auth.role() = 'authenticated');

-- ============================================
-- Cara Testing:
-- ============================================
-- Cek jumlah dokumen:
-- select count(*) from documents;

-- Cek sample data:
-- select id, left(content, 50) as content_preview, metadata 
-- from documents 
-- limit 5;

-- Test similarity search (perlu embedding vector dummy):
-- select * from match_documents(
--   '[0.1, 0.2, ...]'::vector,  -- replace dengan embedding asli
--   5
-- );
