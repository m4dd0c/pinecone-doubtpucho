import { Pinecone } from "@pinecone-database/pinecone";
import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
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
  const body = await request.json().catch(() => ({}));
  const { batchSize = 50, limit, jsonFilePath } = body; // Reduced batch size

  console.log("Starting migration to Pinecone from JSON...");

  // Read JSON data (same as before)
  let questions;

  if (jsonFilePath) {
   const fullPath = path.join(process.cwd(), jsonFilePath);
   const jsonData = fs.readFileSync(fullPath, "utf8");
   questions = JSON.parse(jsonData);
  } else {
   const possiblePaths = [
    path.join(process.cwd(), "questions.json"),
    path.join(process.cwd(), "data", "questions.json"),
    path.join(process.cwd(), "public", "data", "questions.json"),
   ];

   let jsonData;
   for (const filePath of possiblePaths) {
    try {
     jsonData = fs.readFileSync(filePath, "utf8");
     console.log(`Found JSON file at: ${filePath}`);
     break;
    } catch (error) {
     continue;
    }
   }

   if (!jsonData) {
    throw new Error("No JSON file found. Tried: " + possiblePaths.join(", "));
   }

   questions = JSON.parse(jsonData);
  }

  if (!Array.isArray(questions)) {
   throw new Error("JSON data must be an array of questions");
  }

  console.log(`Found ${questions.length} questions in JSON`);

  if (limit) {
   questions = questions.slice(0, limit);
   console.log(`Limited to ${questions.length} questions`);
  }

  // Initialize Pinecone
  const pc = new Pinecone({
   apiKey: process.env.PINECONE_API_KEY,
  });

  const index = pc.index("cat-questions");

  let batch = [];
  let processedCount = 0;
  let errorCount = 0;
  let skippedCount = 0;

  console.log("Processing documents...");

  for (const doc of questions) {
   let docId;
   try {
    // FIXED: Properly extract document ID
    if (doc._id && typeof doc._id === "object" && doc._id.$oid) {
     docId = doc._id.$oid;
    } else if (doc._id) {
     docId = String(doc._id);
    } else if (doc.id) {
     docId = String(doc.id);
    } else {
     console.warn(
      `Skipping document: No valid ID found`,
      JSON.stringify(doc._id || doc.id).substring(0, 100)
     );
     skippedCount++;
     continue;
    }

    // Use cleanedQuestion for embedding, fallback to question if needed
    const textForEmbedding = doc.cleanedQuestion || doc.question;

    if (!textForEmbedding) {
     console.warn(`Skipping document ${docId}: No question text found`);
     skippedCount++;
     continue;
    }

    // Get embedding with improved error handling
    const embedding = await getEmbedding(textForEmbedding);

    // Handle case where Hugging Face returns nested arrays
    const embeddingVector = Array.isArray(embedding[0])
     ? embedding[0]
     : embedding;

    // Prepare vector for Pinecone
    const vector = {
     id: docId,
     values: embeddingVector,
     metadata: {
      subject: doc.subject || "unknown",
      topic: doc.topic || "unknown",
      course: doc.course || "unknown",
      cleanedQuestion: (doc.cleanedQuestion || doc.question)
       .replace(/<[^>]*>/g, "")
       .substring(0, 500),
     },
    };

    batch.push(vector);
    processedCount++;

    // Upsert in smaller batches
    if (batch.length >= batchSize) {
     await index.upsert(batch);
     console.log(
      `✅ Processed ${processedCount} documents (${errorCount} errors, ${skippedCount} skipped)`
     );
     batch = [];

     // Longer delay to avoid rate limiting
     await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    // Add delay every few documents
    if (processedCount % 5 === 0) {
     await new Promise((resolve) => setTimeout(resolve, 200));
    }
   } catch (error) {
    console.error(
     `❌ Error processing document ${docId || "[unknown]"}:`,
     error
    );
    errorCount++;

    // Skip problematic documents and continue
    continue;
   }
  }

  // Process remaining batch
  if (batch.length > 0) {
   await index.upsert(batch);
   console.log(`✅ Final batch: ${batch.length} vectors`);
  }

  console.log("🎉 Migration completed successfully");

  return NextResponse.json({
   success: true,
   message: "Migration completed successfully",
   stats: {
    totalDocuments: questions.length,
    totalProcessed: processedCount,
    errors: errorCount,
    skipped: skippedCount,
    finalBatchSize: batch.length,
   },
  });
 } catch (error) {
  console.error("Migration error:", error);
  return NextResponse.json(
   {
    success: false,
    error: error.message,
    stack: process.env.NODE_ENV === "development" ? error.stack : undefined,
   },
   { status: 500 }
  );
 }
}
