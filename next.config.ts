import type { NextConfig } from 'next';

const [repositoryOwner = '', repositoryName = ''] =
  process.env.GITHUB_REPOSITORY?.split('/') ?? [];
const isGitHubPagesBuild = process.env.GITHUB_PAGES === 'true';
const isAccountSite =
  repositoryName.toLowerCase() ===
  `${repositoryOwner.toLowerCase()}.github.io`;
const pagesAssetPrefix =
  isGitHubPagesBuild && repositoryName && !isAccountSite
    ? `/${repositoryName}`
    : '';

const nextConfig: NextConfig = {
  output: 'export',
  trailingSlash: true,
  assetPrefix: pagesAssetPrefix || undefined,
};

export default nextConfig;
