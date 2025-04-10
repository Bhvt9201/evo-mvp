import { useEffect, useState } from "react";
import Head from "next/head";

export default function Home() {
  const [signal, setSignal] = useState<any>(null);

  useEffect(() => {
    // Local backend during development
    fetch("http://localhost:4000/api/signal")
      .then((res) => res.json())
      .then((data) => {
        console.log("Signal received:", data);
        setSignal(data);
      })
      .catch((err) => console.error("Error fetching signal:", err));
  }, []);

  if (!signal) return <div>Loading intelligent content...</div>;

  const isHindi = signal.deviceInfo.language?.includes("hi");
  const isSpanish = signal.deviceInfo.language?.includes("es");
  const isMacUser = signal.deviceInfo.userAgent?.includes("Macintosh");

  const isAgustin =
    isSpanish && signal.deviceInfo.userAgent?.includes("Windows");

  return (
    <div style={{ textAlign: "center", padding: "50px" }}>
      <Head>
        <title>Evo - AI Powered Websites</title>
      </Head>

      <h1>Welcome to EVO 🧠</h1>
      <p>Your browser: {signal.deviceInfo.userAgent}</p>
      <p>Your preferred language: {signal.deviceInfo.language}</p>

      {isAgustin ? (
        <p>¡Hola Agustín! Bienvenido a tu creación mágica 🪄✨</p>
      ) : isHindi ? (
        <p>आपका स्वागत है! (Welcome in Hindi)</p>
      ) : isSpanish ? (
        <p>¡Bienvenido a Evo, amigo! (Welcome in Spanish) 🇪🇸</p>
      ) : (
        <p>Welcome, awesome human 👋</p>
      )}

      {isMacUser ? (
        <p>You're using a Mac. Want to see Mac-friendly shoes? 🍎👟</p>
      ) : (
        <p>Here's something tailored for your device 🛍️</p>
      )}
    </div>
  );
}
