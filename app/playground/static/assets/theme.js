// Applies a saved colour theme before first paint, so the page never flashes the wrong one.
(function () {
  try {
    var saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") {
      document.documentElement.dataset.theme = saved;
    }
  } catch (e) {
    /* storage can be blocked; the system theme then applies */
  }
})();
