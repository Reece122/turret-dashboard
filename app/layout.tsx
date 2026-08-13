import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AutoTurret",
  description: "Real-time object-tracking turret dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
