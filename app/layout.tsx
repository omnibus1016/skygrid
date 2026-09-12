import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SKYGRID | 군집 무인기 정찰 경로 최적화 플랫폼',
  description: '군집 무인기 정찰 경로 최적화·임무 모의·비행 데이터 검증 플랫폼',
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
