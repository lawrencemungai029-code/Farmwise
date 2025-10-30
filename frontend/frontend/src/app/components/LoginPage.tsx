"use client";
import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "./ui/input";
import FarmerIllustration from "./FarmerIllustration";
import users from "@/data/users.json";
import { motion, AnimatePresence } from "framer-motion";

export default function LoginPage({ onLogin }: { onLogin: (farmer: any) => void }) {
  const [id, setId] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    setTimeout(() => {
      const user = users.find((u) => u.id === id.trim());
      if (user) {
        onLogin({
          name: user.name,
          id: user.id,
          age: 40,
          region: "Nairobi",
          farm_size: 2.5,
          loan_purpose: "inputs",
          disability: 0,
          group_membership: 1,
        });
      } else {
        setError("Invalid ID number, please try again.");
      }
      setLoading(false);
    }, 700);
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-green-100 to-emerald-200 flex flex-col items-center justify-center p-6">
      <motion.div initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }} className="flex flex-col items-center w-full">
        <FarmerIllustration className="w-28 h-28 mb-4" />
        <h1 className="text-3xl font-bold text-green-900 mb-2 text-center drop-shadow">Welcome to SmartFarm Support System</h1>
        <p className="text-lg text-green-800 mb-6 text-center max-w-md">Access credit, monitor progress, and receive personalized support.</p>
        <Card className="w-full max-w-md shadow-xl bg-white/90">
          <CardHeader>
            <CardTitle>Login</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleLogin} className="flex flex-col gap-4">
              <label htmlFor="id" className="font-medium text-green-900">Enter your ID Number</label>
              <Input
                id="id"
                type="text"
                value={id}
                onChange={(e) => setId(e.target.value)}
                placeholder="e.g. 12345"
                className="bg-green-50 border-green-200 focus:border-green-400 focus:ring-green-300"
                autoFocus
                required
              />
              <Button type="submit" className="w-full mt-2" disabled={loading}>
                {loading ? "Logging in..." : "Login"}
              </Button>
              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 10 }}
                    className="text-red-600 text-center font-semibold"
                  >
                    {error}
                  </motion.div>
                )}
              </AnimatePresence>
              <div className="text-xs text-green-700 mt-2 text-center">
                Demo users:<br />
                12345 (John Mwangi), 67890 (Mary Njeri)
              </div>
            </form>
          </CardContent>
        </Card>
      </motion.div>
    </main>
  );
}
