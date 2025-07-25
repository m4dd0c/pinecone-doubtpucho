import { Pinecone } from "@pinecone-database/pinecone";
import { NextResponse } from "next/server";
import { pipeline } from "@xenova/transformers";

// Initialize once at the top of your file (outside the function)
let embedder = null;

async function initEmbedder() {
 if (!embedder) {
  console.log(
   "Loading local embedding model (this may take a minute on first run)..."
  );
  embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
 }
 return embedder;
}

async function getEmbedding(text) {
 const cleanText = text
  .replace(/<[^>]*>/g, "")
  .replace(/\s+/g, " ")
  .trim();

 if (!cleanText || cleanText.length < 3) {
  throw new Error("Text too short after cleaning");
 }

 const embedder = await initEmbedder();
 const output = await embedder(cleanText, { pooling: "mean", normalize: true });
 return Array.from(output.data);
}

export async function POST(request) {
 try {
  const { query, subject, topic, course, topK = 10 } = await request.json();

  if (!query || query.trim().length === 0) {
   return NextResponse.json({ error: "Query is required" }, { status: 400 });
  }

  // 1. Get embedding for user query using local model
  const queryEmbedding = await getEmbedding(query);

  // 2. Search in Pinecone
  const pc = new Pinecone({
   apiKey: process.env.PINECONE_API_KEY,
  });

  const index = pc.index("cat-questions");

  // Build filter for metadata
  const filter = {};
  if (subject) filter.subject = subject;
  if (topic) filter.topic = topic;
  if (course) filter.course = course;

  const searchResults = await index.query({
   vector: queryEmbedding,
   topK,
   filter: Object.keys(filter).length > 0 ? filter : undefined,
   includeMetadata: true,
  });

  console.log(`Found ${searchResults.matches.length} matches`);

  // 3. Return Pinecone results directly (no MongoDB lookup)
  return NextResponse.json({
   success: true,
   matches: searchResults.matches,
   total: searchResults.matches.length,
   searchQuery: query,
   filters: filter,
  });
 } catch (error) {
  console.error("Search error:", error);
  return NextResponse.json(
   {
    success: false,
    error: error.message,
   },
   { status: 500 }
  );
 }
}
