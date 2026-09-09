import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SKYGRID | UAS Mission Continuity System',
  description:
    '강화학습 기반 다중 무인기 정찰 임무 시뮬레이션 및 비행 검증 플랫폼',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko" className="dark">
      <body className="antialiased">{children}</body>
    </html>
  );
}
