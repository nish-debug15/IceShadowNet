/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  basePath: '/IceShadowNet/dashboard',
  assetPrefix: '/IceShadowNet/dashboard',
  images: {
    unoptimized: true,
  },
  trailingSlash: true,
};

export default nextConfig;
