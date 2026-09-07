const paths = {
  layers: '<path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
  camera:
    '<path d="M8 5 6 8H3v12h18V8h-3l-2-3Z"/><circle cx="12" cy="13" r="4"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',
  minus: '<path d="M5 12h14"/>',
  shuffle:
    '<path d="M3 6h3c5 0 7 12 12 12h3m-4-4 4 4-4 4M3 18h3c2 0 4-2 6-6s4-6 6-6h3m-4-4 4 4-4 4"/>',
  spark:
    '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5ZM20 2v4m-2-2h4"/>',
  mountain: '<path d="m2 20 8-16 5 10 3-5 5 11ZM7 10l3 2 3-2"/>',
  down: '<path d="M12 3v13m-5-5 5 5 5-5M4 21h16"/>',
  leaf: '<path d="M20 3C7 2 3 7 4 14c1 5 8 6 12 2s4-13 4-13ZM3 21 15 9"/>',
  drop: '<path d="M12 2c-2 4-8 9-8 13a8 8 0 0 0 16 0c0-4-6-9-8-13ZM8 15c0 2 1 3 3 3"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 1v2m0 18v2M1 12h2m18 0h2M4 4l2 2m12 12 2 2M20 4l-2 2M6 18l-2 2"/>',
  hand: '<path d="M8 13V7a2 2 0 0 1 4 0v6-8a2 2 0 0 1 4 0v8-6a2 2 0 0 1 4 0v9c0 5-3 6-7 6-3 0-4-1-6-4l-4-5a2 2 0 0 1 3-2l2 2Z"/>',
  play: '<path d="m7 3 14 9-14 9Z"/>',
  branch:
    '<circle cx="6" cy="4" r="2"/><circle cx="6" cy="20" r="2"/><circle cx="18" cy="5" r="2"/><path d="M6 6v12m0-6h5a7 7 0 0 0 7-5"/>',
  refresh:
    '<path d="M20 7v5h-5M4 17v-5h5M19 11a7 7 0 0 0-12-6L4 8m1 5a7 7 0 0 0 12 6l3-3"/>',
  download: '<path d="M12 2v13m-5-5 5 5 5-5M3 16v5h18v-5"/>',
  upload: '<path d="M12 15V2m-5 5 5-5 5 5M3 16v5h18v-5"/>',
  orbit:
    '<circle cx="12" cy="12" r="3"/><ellipse cx="12" cy="12" rx="11" ry="5" transform="rotate(-35 12 12)"/>',
  walk: '<circle cx="13" cy="4" r="2"/><path d="m8 21 3-7 3 3v5M4 12l4-4h5l3 5h5M11 8v6"/>',
  home: '<path d="m3 11 9-8 9 8M6 10v11h12V10M10 21v-7h4v7"/>',
  expand: '<path d="M3 9V3h6m6 0h6v6M3 15v6h6m6 0h6v-6"/>',
  close: '<path d="m5 5 14 14M19 5 5 19"/>',
};
export const icon = (name) =>
  `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[name] || paths.info}</svg>`;
export function mountIcons(root = document) {
  root.querySelectorAll("[data-icon]").forEach((el) => {
    el.innerHTML = icon(el.dataset.icon);
  });
}
