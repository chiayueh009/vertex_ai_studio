// app/page.tsx (Next.js 13+)
"use client";
import { useState } from "react";

export default function Page() {
  const [q, setQ] = useState("");
  const [a, setA] = useState("");

  const ask = async () => {
    setA("查詢中…");
    const res = await fetch(process.env.NEXT_PUBLIC_API_URL + "/chat", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ query: q })
    });
    const data = await res.json();
    setA(data.answer || "沒有回覆");
  };

  return (
    <div style={{maxWidth:720, margin:"40px auto"}}>
      <h1>企業智能客服（FAQ + 法規檢索）</h1>
      <textarea value={q} onChange={e=>setQ(e.target.value)} rows={6} style={{width:"100%"}}/>
      <button onClick={ask}>送出</button>
      <pre style={{whiteSpace:"pre-wrap", background:"#f6f6f6", padding:12}}>{a}</pre>
    </div>
  );
}
