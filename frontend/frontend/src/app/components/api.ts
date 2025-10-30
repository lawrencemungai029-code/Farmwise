export async function fetchCreditScore(farmer: any) {
  const res = await fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(farmer),
  });
  if (!res.ok) throw new Error("Failed to fetch credit score");
  return res.json();
}
