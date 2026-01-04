const CONFIG = {
  /* Animations to use */
  animations: ["torchGlow1", "torchGlow2", "torchGlow3", "torchGlow4"],
  /* Number of dots to start glowing immediately on load */
  initialGlowingDots: 20,
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
  const container = document.getElementById('glowing-logo-container');
  if (!container) return;

  fetch('/assets/medias/pytorch-pointcloud-logo.svg')
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
  const svg = document.querySelector("#glowing-logo-container svg");
  const circleElements = svg.querySelectorAll("circle.dot");
  const dots = [];

  // 1. Setup Dots & Cache Coordinates
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
  startFullBloomCycle(); // <--- NEW: Start the full bloom timer

  // 2. The Global Mouse Tracker
  svg.addEventListener('mousemove', (e) => {
    const rect = svg.getBoundingClientRect();
    const viewBox = svg.viewBox.baseVal;
    
    // ViewBox/Dimension Safety Checks
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
  });

  // 3. Core Glow Function
  function glowDot(dot, isHover = false) {
    if (dot.isGlowing) return;

    dot.isGlowing = true;
    const originalCircle = dot.element;
    const clone = originalCircle.cloneNode(true);
    
    clone.classList.remove("dot");
    clone.classList.add("glowing-clone"); 

    svg.appendChild(clone);

    const animName = CONFIG.animations[Math.floor(Math.random() * CONFIG.animations.length)];
    
    // Logic: 
    // If Hover: 1.5s
    // If Bloom/Random: Random between min/max duration
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

  // 4. Auto-Animation Helpers
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

  /* --- NEW FUNCTION --- */
  function startFullBloomCycle() {
    // Determine random time until the next "Full Bloom" event
    const nextBloomTime = Math.random() * (CONFIG.maxBloomInterval - CONFIG.minBloomInterval) + CONFIG.minBloomInterval;

    setTimeout(() => {
        // Trigger every dot
        dots.forEach((dot) => {
            // We pass false to use the longer, organic duration. 
            // Pass true if you want the bloom to happen quickly/snappily.
            glowDot(dot, false); 
        });
        
        // Schedule the next one
        startFullBloomCycle();
    }, nextBloomTime);
  }
}

// function initializeGlowAnimation() {
//   const svg = document.querySelector("#glowing-logo-container svg");
//   const circleElements = svg.querySelectorAll("circle.dot");
//   const dots = [];

//   // 1. Setup Dots & Cache Coordinates
//   // We cache cx/cy so we don't have to read the DOM on every mouse move (performance)
//   circleElements.forEach((circle) => {
//     // Force pointer events off the visual dots so they don't block mouse movements
//     circle.style.pointerEvents = "none"; 
    
//     dots.push({
//       element: circle,
//       isGlowing: false,
//       cx: parseFloat(circle.getAttribute("cx")),
//       cy: parseFloat(circle.getAttribute("cy")),
//       r: parseFloat(circle.getAttribute("r")),
//     });
//   });

//   // Start background animations
//   startInitialGlowingDots();
//   startContinuousChains();

//   // 2. The Global Mouse Tracker (The "Flashlight" logic)
//   svg.addEventListener('mousemove', (e) => {
//     // Get SVG screen dimensions
//     const rect = svg.getBoundingClientRect();
    
//     // Calculate mapping from Screen Pixels -> SVG Units
//     // (This handles if the SVG is resized via CSS)
//     const viewBox = svg.viewBox.baseVal;
//     // Fallback if viewBox isn't explicitly set in the SVG file
//     const vbWidth = viewBox && viewBox.width ? viewBox.width : rect.width;
//     const vbHeight = viewBox && viewBox.height ? viewBox.height : rect.height;
//     const vbX = viewBox ? viewBox.x : 0;
//     const vbY = viewBox ? viewBox.y : 0;

//     const scaleX = vbWidth / rect.width;
//     const scaleY = vbHeight / rect.height;

//     // Mouse Position in SVG Coordinates
//     const mouseX = (e.clientX - rect.left) * scaleX + vbX;
//     const mouseY = (e.clientY - rect.top) * scaleY + vbY;

//     // Check every dot to see if it's inside the "activationRadius"
//     dots.forEach(dot => {
//       // Euclidean distance formula: sqrt(dx^2 + dy^2)
//       const dx = mouseX - dot.cx;
//       const dy = mouseY - dot.cy;
//       // Optimization: Compare squared distances to avoid expensive Math.sqrt()
//       const distSquared = dx * dx + dy * dy;
//       const radiusSquared = CONFIG.activationRadius * CONFIG.activationRadius;

//       if (distSquared < radiusSquared) {
//         glowDot(dot, true);
//       }
//     });
//   });

//   // 3. Core Glow Function
//   function glowDot(dot, isHover = false) {
//     if (dot.isGlowing) return;

//     dot.isGlowing = true;
//     const originalCircle = dot.element;
//     const clone = originalCircle.cloneNode(true);
    
//     clone.classList.remove("dot");
//     clone.classList.add("glowing-clone"); 

//     // Append to SVG to ensure it sits on top of everything
//     svg.appendChild(clone);

//     const animName = CONFIG.animations[Math.floor(Math.random() * CONFIG.animations.length)];
    
//     const duration = isHover ? 1.5 : (
//       Math.random() * (CONFIG.maxDuration - CONFIG.minDuration) +
//       CONFIG.minDuration
//     ).toFixed(2);

//     clone.style.animation = `${animName} ${duration}s ease-in-out forwards`;

//     setTimeout(() => {
//       clone.remove();
//       dot.isGlowing = false;
//     }, parseFloat(duration) * 1000);
//   }

//   // 4. Auto-Animation Helpers
//   function selectRandomDot() {
//     const availableDots = dots.filter((d) => !d.isGlowing);
//     if (availableDots.length === 0)
//       return dots[Math.floor(Math.random() * dots.length)];
//     return availableDots[Math.floor(Math.random() * availableDots.length)];
//   }

//   function startInitialGlowingDots() {
//     for (let i = 0; i < CONFIG.initialGlowingDots; i++) {
//       setTimeout(() => {
//         const dot = selectRandomDot();
//         glowDot(dot, false);
//       }, Math.random() * 600); 
//     }
//   }

//   function startContinuousChains() {
//     function runChain() {
//       const dot = selectRandomDot();
//       glowDot(dot, false);
//       const nextGlow = Math.random() * (CONFIG.maxInterval - CONFIG.minInterval) + CONFIG.minInterval;
//       setTimeout(runChain, nextGlow);
//     }

//     for (let i = 0; i < CONFIG.simultaneousChains; i++) {
//       setTimeout(runChain, Math.random() * CONFIG.chainStaggerDelay);
//     }
//   }
// }


// V2 IS OK
// function initializeGlowAnimation() {
//   const svg = document.querySelector("#glowing-logo-container svg");
//   const circleElements = svg.querySelectorAll("circle.dot");
//   const dots = [];
//   const animations = ["flameGlow1", "flameGlow2", "flameGlow3", "flameGlow4"];

//   // Function to make a dot glow with organic variations
//   // Moved up so it can be used in the event listener definition
//   function glowDot(dot, isHover = false) {
//     if (dot.isGlowing) return;

//     dot.isGlowing = true;
//     const circle = dot.element;

//     // Bring glowing circle to front
//     circle.parentNode.appendChild(circle);

//     // Random animation variant
//     const animName = animations[Math.floor(Math.random() * animations.length)];

//     // Determine duration: Use random config for auto, or a fixed snappy duration for hover
//     let duration;
//     if (isHover) {
//        // Make hovers slightly faster/snappier (e.g. 1.5s) or use CONFIG.minDuration
//        duration = 1.5; 
//     } else {
//        duration = (
//         Math.random() * (CONFIG.maxDuration - CONFIG.minDuration) +
//         CONFIG.minDuration
//       ).toFixed(2);
//     }

//     circle.classList.add("glowing");
//     circle.style.animation = `${animName} ${duration}s ease-in-out`;

//     setTimeout(() => {
//       circle.style.animation = "";
//       circle.classList.remove("glowing");
//       dot.isGlowing = false;
//     }, parseFloat(duration) * 1000);
//   }

//   // Store information and attach listeners
//   circleElements.forEach((circle) => {
//     const r = parseFloat(circle.getAttribute("r"));
    
//     const dot = {
//       element: circle,
//       isGlowing: false,
//       baseRadius: r,
//     };
    
//     dots.push(dot);

//     // --- NEW CODE: Add Hover Listener ---
//     circle.addEventListener('mouseenter', () => {
//       // Pass 'true' to indicate this is a hover event
//       glowDot(dot, true);
//     });
//   });

//   // Weighted random selection - prefer dots not recently glowed
//   function selectRandomDot() {
//     const availableDots = dots.filter((d) => !d.isGlowing);
//     if (availableDots.length === 0)
//       return dots[Math.floor(Math.random() * dots.length)];
//     return availableDots[Math.floor(Math.random() * availableDots.length)];
//   }

//   // Start glowing chains with organic timing
//   function startGlowingChain() {
//     const dot = selectRandomDot();
//     glowDot(dot, false); // Pass false for auto-animation

//     const nextGlow =
//       Math.random() * (CONFIG.maxInterval - CONFIG.minInterval) +
//       CONFIG.minInterval;
//     setTimeout(startGlowingChain, nextGlow);
//   }

//   // Start multiple independent chains
//   for (let i = 0; i < CONFIG.simultaneousChains; i++) {
//     setTimeout(startGlowingChain, Math.random() * CONFIG.chainStaggerDelay);
//   }
// }


// V1 IS OK


// function initializeGlowAnimation() {
//   const svg = document.querySelector("#glowing-logo-container svg");
//   const circleElements = svg.querySelectorAll("circle.dot");
//   const dots = [];
//   const animations = ["flameGlow1", "flameGlow2", "flameGlow3", "flameGlow4"];

//   // Store information about each existing circle
//   circleElements.forEach((circle) => {
//     const r = parseFloat(circle.getAttribute("r"));
//     dots.push({
//       element: circle,
//       isGlowing: false,
//       baseRadius: r,
//     });
//   });

//   // Function to make a dot glow with organic variations
//   function glowDot(dot) {
//     if (dot.isGlowing) return;

//     dot.isGlowing = true;
//     const circle = dot.element;

//     // Bring glowing circle to front
//     circle.parentNode.appendChild(circle);

//     // Random animation variant
//     const animName = animations[Math.floor(Math.random() * animations.length)];

//     // Random duration using configured range
//     const duration = (
//       Math.random() * (CONFIG.maxDuration - CONFIG.minDuration) +
//       CONFIG.minDuration
//     ).toFixed(2);

//     circle.classList.add("glowing");
//     circle.style.animation = `${animName} ${duration}s ease-in-out`;

//     setTimeout(() => {
//       circle.style.animation = "";
//       circle.classList.remove("glowing");
//       dot.isGlowing = false;
//     }, parseFloat(duration) * 1000);
//   }

//   // Weighted random selection - prefer dots not recently glowed
//   function selectRandomDot() {
//     const availableDots = dots.filter((d) => !d.isGlowing);
//     if (availableDots.length === 0)
//       return dots[Math.floor(Math.random() * dots.length)];
//     return availableDots[Math.floor(Math.random() * availableDots.length)];
//   }

//   // Start glowing chains with organic timing
//   function startGlowingChain() {
//     const dot = selectRandomDot();
//     glowDot(dot);

//     // More organic, varied intervals using configured range
//     const nextGlow =
//       Math.random() * (CONFIG.maxInterval - CONFIG.minInterval) +
//       CONFIG.minInterval;
//     setTimeout(startGlowingChain, nextGlow);
//   }

//   // Start multiple independent chains with staggered starts
//   for (let i = 0; i < CONFIG.simultaneousChains; i++) {
//     setTimeout(startGlowingChain, Math.random() * CONFIG.chainStaggerDelay);
//   }
// }
