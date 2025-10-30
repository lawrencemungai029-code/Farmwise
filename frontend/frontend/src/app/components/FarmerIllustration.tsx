import * as React from "react";

export default function FarmerIllustration({ className = "w-24 h-24" }) {
  return (
    <svg viewBox="0 0 64 64" fill="none" className={className}>
      <circle cx="32" cy="32" r="32" fill="#FDE68A" />
      <ellipse cx="32" cy="44" rx="18" ry="10" fill="#A7F3D0" />
      <ellipse cx="32" cy="28" rx="10" ry="12" fill="#FBBF24" />
      <ellipse cx="32" cy="24" rx="6" ry="7" fill="#F59E42" />
      <ellipse cx="32" cy="22" rx="3" ry="3.5" fill="#fff" />
      <rect x="28" y="38" width="8" height="10" rx="3" fill="#F59E42" />
      <ellipse cx="32" cy="48" rx="4" ry="2" fill="#A7F3D0" />
    </svg>
  );
}
