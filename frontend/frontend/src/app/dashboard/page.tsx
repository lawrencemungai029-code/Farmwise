"use client";
import React, { useEffect, useState } from "react";
import Dashboard from "../components/Dashboard";
import JourneyPage from "../components/JourneyPage";
import { fetchCreditScore } from "../components/api";
import { useRouter } from "next/navigation";

export default function DashboardPage() {
  const [result, setResult] = useState<any>(null);
  const [stage, setStage] = useState<"dashboard" | "journey">("dashboard");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    // For demo, use a default farmer (John Mwangi)
    const farmer = {
      name: "John Mwangi",
      id: "12345",
      age: 40,
      region: "Nairobi",
      farm_size: 2.5,
      loan_purpose: "inputs",
      disability: 0,
      group_membership: 1,
    };
    fetchCreditScore(farmer)
      .then(setResult)
      .catch((e) => setError(e.message || "Failed to fetch credit score"))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="flex min-h-screen items-center justify-center">Loading...</div>;
  if (error) return <div className="flex min-h-screen items-center justify-center text-red-600">{error}</div>;
  if (stage === "dashboard") return <Dashboard result={result} onContinue={() => setStage("journey")} />;
  if (stage === "journey") return <JourneyPage onFinish={() => setStage("dashboard")} />;
  return null;
}
