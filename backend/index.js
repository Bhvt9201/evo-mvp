const express = require("express");
const cors = require("cors");

const app = express();
const port = process.env.PORT || 4000;

app.use(cors());
app.use(express.json());

app.get("/api/signal", (req, res) => {
  try {
    const userAgent = req.get("User-Agent");
    const acceptLanguage = req.get("Accept-Language");

    // Get timezone safely
    let timezone = "unknown";
    try {
      timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    } catch (e) {
      console.warn("Could not detect timezone");
    }

    const deviceInfo = {
      userAgent,
      language: acceptLanguage,
      timezone,
    };

    console.log("Device info sent:", deviceInfo);

    res.json({
      message: "Signal received!",
      deviceInfo,
    });
  } catch (error) {
    console.error("🔥 Error in /api/signal:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.listen(port, () => {
  console.log(`✅ Backend server is running on http://localhost:${port}`);
});
