const express = require("express");
const cors = require("cors");

const app = express();
app.use(cors()); // Allow frontend to access backend
app.use(express.json()); // Enable JSON parsing

// Define the test API route
app.get("/api/test", (req, res) => {
  res.json({ message: "Backend is working!" });
});

// Start the server
app.listen(4000, () => {
  console.log("Server running on port 4000");
});

