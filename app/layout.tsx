import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SKYGRID | 군집 무인기 정찰 임무통제체계',
  description:
    '강화학습 기반 군집 무인기 정찰 임무계획·시뮬레이션·비행 검증 체계',
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
