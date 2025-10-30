"use client";
import React from "react";
import LoginPage from "./components/LoginPage";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();
  return (
    <LoginPage onLogin={() => router.push("/dashboard")} />
  );
}
