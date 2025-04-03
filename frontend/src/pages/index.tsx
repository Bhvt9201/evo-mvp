import { useEffect, useState } from "react";
import Head from "next/head";

export default function Home() {
  const [data, setData] = useState<string | null>(null);

  useEffect(() => {
    fetch("http://localhost:4000/api/test")
      .then((res) => res.json())
      .then((data) => setData(data.message))
      .catch((err) => console.error("Error fetching data:", err));
  }, []);

  return (
    <div style={{ textAlign: "center", padding: "50px" }}>
      <Head>
        <title>Evo - AI Powered Websites</title>
      </Head>
      <h1>Welcome to Evo</h1>
      <p>
        Evo automatically updates, optimizes, and improves websites without
        manual effort.
      </p>  

      {/* ✅ Add this line to display the backend response */}
      <p>Backend Response: {data ? data : "Loading..."}</p>    

      <button   
        style={{
          padding: "10px 20px",
          fontSize: "16px", 
          cursor: "pointer",
          backgroundColor: "black",
          color: "white",
          border: "none",
          borderRadius: "5px",
          marginTop: "20px",
        }}
      >
        Get Started
      </button>
    </div>
  );
}

