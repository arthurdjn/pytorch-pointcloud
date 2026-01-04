const CONFIG = {
  /* Asset config */
  containerId: "glowing-logo-container",
  svgPath: "/assets/medias/pytorch-pointcloud-logo.svg",
  /* Animations to use */
  animations: ["torchGlow1", "torchGlow2", "torchGlow3", "torchGlow4"],
  /* Number of dots to start glowing immediately on load */
  initialGlowingDots: 25,
  /* Radius around the mouse that triggers dots (in SVG units) */
  activationRadius: 25, 
  /* Minimum glow duration (s) */
  minDuration: 7,
  /* Maximum glow duration (s) */
  maxDuration: 10,
  /* Minimum time between glows (ms) */
  minInterval: 1000,
  /* Maximum time between glows (ms) */
  maxInterval: 2000,
  /* Number of independent glow chains */
  simultaneousChains: 5,
  /* Max delay between chain starts (ms) */
  chainStaggerDelay: 6000,
  /* Minimum time between full blooms (ms) */
  minBloomInterval: 10000, 
  /* Maximum time between full blooms (ms) */
  maxBloomInterval: 20000,
  
};

document.addEventListener('DOMContentLoaded', function () {
  const container = document.getElementById(CONFIG.containerId);
  if (!container) return;

  fetch(CONFIG.svgPath)
    .then(response => response.text())
    .then(svgContent => {
      container.innerHTML = svgContent;
      initializeGlowAnimation();
    })
    .catch(error => {
      console.error("Error loading SVG:", error);
    });
});

function initializeGlowAnimation() {
  const svg = document.querySelector(`#${CONFIG.containerId} svg`);
  const circleElements = svg.querySelectorAll("circle.dot");

  // Dots is an array of objects with the following properties:
  // - element: the circle element
  // - isGlowing: a boolean indicating if the dot is currently glowing
  // - cx: the x coordinate of the dot
  // - cy: the y coordinate of the dot
  // - r: the radius of the dot
  const dots = [];

  circleElements.forEach((circle) => {
    circle.style.pointerEvents = "none"; 
    
    dots.push({
      element: circle,
      isGlowing: false,
      cx: parseFloat(circle.getAttribute("cx")),
      cy: parseFloat(circle.getAttribute("cy")),
      r: parseFloat(circle.getAttribute("r")),
    });
  });

  // Start background animations
  startInitialGlowingDots();
  startContinuousChains();
  startFullBloomCycle();
  svg.addEventListener('mousemove', handleDotHover);

  function handleDotHover(e) {
    const rect = svg.getBoundingClientRect();
    const viewBox = svg.viewBox.baseVal;
    
    // ViewBox/Dimension Safety Checks (handles if the SVG is resized via CSS)
    const vbWidth = viewBox && viewBox.width ? viewBox.width : rect.width;
    const vbHeight = viewBox && viewBox.height ? viewBox.height : rect.height;
    const vbX = viewBox ? viewBox.x : 0;
    const vbY = viewBox ? viewBox.y : 0;

    const scaleX = vbWidth / rect.width;
    const scaleY = vbHeight / rect.height;

    const mouseX = (e.clientX - rect.left) * scaleX + vbX;
    const mouseY = (e.clientY - rect.top) * scaleY + vbY;

    dots.forEach(dot => {
      const dx = mouseX - dot.cx;
      const dy = mouseY - dot.cy;
      const distSquared = dx * dx + dy * dy;
      const radiusSquared = CONFIG.activationRadius * CONFIG.activationRadius;

      if (distSquared < radiusSquared) {
        glowDot(dot, true);
      }
    });
  }

  function glowDot(dot, isHover = false) {
    if (dot.isGlowing) return;

    dot.isGlowing = true;
    const originalCircle = dot.element;
    const clone = originalCircle.cloneNode(true);
    
    clone.classList.remove("dot");
    clone.classList.add("glowing-clone"); 

    svg.appendChild(clone);

    const animName = CONFIG.animations[Math.floor(Math.random() * CONFIG.animations.length)];
    
    // If the dot is being hovered, use a shorter duration (1.5s)
    const duration = isHover ? 1.5 : (
      Math.random() * (CONFIG.maxDuration - CONFIG.minDuration) +
      CONFIG.minDuration
    ).toFixed(2);

    clone.style.animation = `${animName} ${duration}s ease-in-out forwards`;

    setTimeout(() => {
      clone.remove();
      dot.isGlowing = false;
    }, parseFloat(duration) * 1000);
  }

  function selectRandomDot() {
    const availableDots = dots.filter((d) => !d.isGlowing);
    if (availableDots.length === 0)
      return dots[Math.floor(Math.random() * dots.length)];
    return availableDots[Math.floor(Math.random() * availableDots.length)];
  }

  function startInitialGlowingDots() {
    for (let i = 0; i < CONFIG.initialGlowingDots; i++) {
      setTimeout(() => {
        const dot = selectRandomDot();
        glowDot(dot, false);
      }, Math.random() * 600); 
    }
  }

  function startContinuousChains() {
    function runChain() {
      const dot = selectRandomDot();
      glowDot(dot, false);
      const nextGlow = Math.random() * (CONFIG.maxInterval - CONFIG.minInterval) + CONFIG.minInterval;
      setTimeout(runChain, nextGlow);
    }

    for (let i = 0; i < CONFIG.simultaneousChains; i++) {
      setTimeout(runChain, Math.random() * CONFIG.chainStaggerDelay);
    }
  }

  function startFullBloomCycle() {
    const nextBloomTime = Math.random() * (CONFIG.maxBloomInterval - CONFIG.minBloomInterval) + CONFIG.minBloomInterval;

    setTimeout(() => {
        dots.forEach((dot) => {
            glowDot(dot, false); 
        });
        
        startFullBloomCycle();
    }, nextBloomTime);
  }
}
