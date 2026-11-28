export const BASE_PATH = '/IceShadowNet/dashboard';

export function withBasePath(path: string): string {
  if (!path) return BASE_PATH;
  if (
    path.startsWith('http://') ||
    path.startsWith('https://') ||
    path.startsWith(BASE_PATH)
  ) {
    return path;
  }
  const clean = path.startsWith('/') ? path : `/${path}`;
  return `${BASE_PATH}${clean}`;
}
