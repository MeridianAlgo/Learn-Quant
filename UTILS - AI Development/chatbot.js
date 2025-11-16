<<<<<<< HEAD
// Gemini API Chatbot in Node.js
// This script allows you to chat with Google's Gemini model from the command line.
// The API key is set directly in the script for demonstration purposes.
// For production, consider using a .env file or environment variable for security.

import readlineSync from 'readline-sync'; // For command-line input
import { GoogleGenAI } from '@google/genai'; // Official Gemini API Node.js SDK

// Set the Gemini API key directly (replace with your own key)
const API_KEY = "YOUR_GEMINI_API_KEY";

// Initialize the Gemini API client
const ai = new GoogleGenAI({ apiKey: API_KEY });

console.log("Gemini Chatbot (type 'exit' to quit)");

// Main chat loop
async function chat() {
  while (true) {
    const userInput = readlineSync.question('You: '); // Get user input
    if (userInput.toLowerCase() === 'exit') break; // Exit condition
    // Send the user input to Gemini and get the response
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash", // Use the high-throughput Gemini model
      contents: userInput
    });
    console.log('Gemini:', response.text); // Print Gemini's response
  }
}

=======
// Gemini API Chatbot in Node.js
// This script allows you to chat with Google's Gemini model from the command line.
// The API key is set directly in the script for demonstration purposes.
// For production, consider using a .env file or environment variable for security.

import readlineSync from 'readline-sync'; // For command-line input
import { GoogleGenAI } from '@google/genai'; // Official Gemini API Node.js SDK

// Set the Gemini API key directly (replace with your own key)
const API_KEY = "YOUR_GEMINI_API_KEY";

// Initialize the Gemini API client
const ai = new GoogleGenAI({ apiKey: API_KEY });

console.log("Gemini Chatbot (type 'exit' to quit)");

// Main chat loop
async function chat() {
  while (true) {
    const userInput = readlineSync.question('You: '); // Get user input
    if (userInput.toLowerCase() === 'exit') break; // Exit condition
    // Send the user input to Gemini and get the response
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash", // Use the high-throughput Gemini model
      contents: userInput
    });
    console.log('Gemini:', response.text); // Print Gemini's response
  }
}

>>>>>>> 8944b09 (Initial commit: Comprehensive Python & JS Finance Utilities for Beginners (API & API-free, with detailed docs))
chat();