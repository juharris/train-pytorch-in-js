
var ortWasm = (() => {
  var _scriptDir = typeof document !== 'undefined' && document.currentScript ? document.currentScript.src : undefined;
  if (typeof __filename !== 'undefined') _scriptDir = _scriptDir || __filename;
  return (
function(ortWasm) {
  ortWasm = ortWasm || {};


var b;
b || (b = typeof ortWasm !== 'undefined' ? ortWasm : {});
var aa = Object.assign, ba, ca;
b.ready = new Promise(function(a, c) {
  ba = a;
  ca = c;
});
Object.getOwnPropertyDescriptor(b.ready, "_OrtInit") || (Object.defineProperty(b.ready, "_OrtInit", {configurable:!0, get:function() {
  n("You are getting _OrtInit on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtInit", {configurable:!0, set:function() {
  n("You are setting _OrtInit on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtCreateSessionOptions") || (Object.defineProperty(b.ready, "_OrtCreateSessionOptions", {configurable:!0, get:function() {
  n("You are getting _OrtCreateSessionOptions on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtCreateSessionOptions", {configurable:!0, set:function() {
  n("You are setting _OrtCreateSessionOptions on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtAddSessionConfigEntry") || (Object.defineProperty(b.ready, "_OrtAddSessionConfigEntry", {configurable:!0, get:function() {
  n("You are getting _OrtAddSessionConfigEntry on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtAddSessionConfigEntry", {configurable:!0, set:function() {
  n("You are setting _OrtAddSessionConfigEntry on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtReleaseSessionOptions") || (Object.defineProperty(b.ready, "_OrtReleaseSessionOptions", {configurable:!0, get:function() {
  n("You are getting _OrtReleaseSessionOptions on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtReleaseSessionOptions", {configurable:!0, set:function() {
  n("You are setting _OrtReleaseSessionOptions on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtCreateSession") || (Object.defineProperty(b.ready, "_OrtCreateSession", {configurable:!0, get:function() {
  n("You are getting _OrtCreateSession on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtCreateSession", {configurable:!0, set:function() {
  n("You are setting _OrtCreateSession on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtReleaseSession") || (Object.defineProperty(b.ready, "_OrtReleaseSession", {configurable:!0, get:function() {
  n("You are getting _OrtReleaseSession on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtReleaseSession", {configurable:!0, set:function() {
  n("You are setting _OrtReleaseSession on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtGetInputCount") || (Object.defineProperty(b.ready, "_OrtGetInputCount", {configurable:!0, get:function() {
  n("You are getting _OrtGetInputCount on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtGetInputCount", {configurable:!0, set:function() {
  n("You are setting _OrtGetInputCount on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtGetOutputCount") || (Object.defineProperty(b.ready, "_OrtGetOutputCount", {configurable:!0, get:function() {
  n("You are getting _OrtGetOutputCount on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtGetOutputCount", {configurable:!0, set:function() {
  n("You are setting _OrtGetOutputCount on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtGetInputName") || (Object.defineProperty(b.ready, "_OrtGetInputName", {configurable:!0, get:function() {
  n("You are getting _OrtGetInputName on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtGetInputName", {configurable:!0, set:function() {
  n("You are setting _OrtGetInputName on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtGetOutputName") || (Object.defineProperty(b.ready, "_OrtGetOutputName", {configurable:!0, get:function() {
  n("You are getting _OrtGetOutputName on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtGetOutputName", {configurable:!0, set:function() {
  n("You are setting _OrtGetOutputName on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtFree") || (Object.defineProperty(b.ready, "_OrtFree", {configurable:!0, get:function() {
  n("You are getting _OrtFree on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtFree", {configurable:!0, set:function() {
  n("You are setting _OrtFree on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtCreateTensor") || (Object.defineProperty(b.ready, "_OrtCreateTensor", {configurable:!0, get:function() {
  n("You are getting _OrtCreateTensor on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtCreateTensor", {configurable:!0, set:function() {
  n("You are setting _OrtCreateTensor on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtGetTensorData") || (Object.defineProperty(b.ready, "_OrtGetTensorData", {configurable:!0, get:function() {
  n("You are getting _OrtGetTensorData on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtGetTensorData", {configurable:!0, set:function() {
  n("You are setting _OrtGetTensorData on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtReleaseTensor") || (Object.defineProperty(b.ready, "_OrtReleaseTensor", {configurable:!0, get:function() {
  n("You are getting _OrtReleaseTensor on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtReleaseTensor", {configurable:!0, set:function() {
  n("You are setting _OrtReleaseTensor on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtCreateRunOptions") || (Object.defineProperty(b.ready, "_OrtCreateRunOptions", {configurable:!0, get:function() {
  n("You are getting _OrtCreateRunOptions on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtCreateRunOptions", {configurable:!0, set:function() {
  n("You are setting _OrtCreateRunOptions on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtAddRunConfigEntry") || (Object.defineProperty(b.ready, "_OrtAddRunConfigEntry", {configurable:!0, get:function() {
  n("You are getting _OrtAddRunConfigEntry on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtAddRunConfigEntry", {configurable:!0, set:function() {
  n("You are setting _OrtAddRunConfigEntry on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtReleaseRunOptions") || (Object.defineProperty(b.ready, "_OrtReleaseRunOptions", {configurable:!0, get:function() {
  n("You are getting _OrtReleaseRunOptions on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtReleaseRunOptions", {configurable:!0, set:function() {
  n("You are setting _OrtReleaseRunOptions on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtRun") || (Object.defineProperty(b.ready, "_OrtRun", {configurable:!0, get:function() {
  n("You are getting _OrtRun on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtRun", {configurable:!0, set:function() {
  n("You are setting _OrtRun on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "_OrtEndProfiling") || (Object.defineProperty(b.ready, "_OrtEndProfiling", {configurable:!0, get:function() {
  n("You are getting _OrtEndProfiling on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "_OrtEndProfiling", {configurable:!0, set:function() {
  n("You are setting _OrtEndProfiling on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(b.ready, "onRuntimeInitialized") || (Object.defineProperty(b.ready, "onRuntimeInitialized", {configurable:!0, get:function() {
  n("You are getting onRuntimeInitialized on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(b.ready, "onRuntimeInitialized", {configurable:!0, set:function() {
  n("You are setting onRuntimeInitialized on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
var da = aa({}, b), ea = "./this.program", fa = "object" === typeof window, ha = "function" === typeof importScripts, ia = "object" === typeof process && "object" === typeof process.versions && "string" === typeof process.versions.node, ja = !fa && !ia && !ha;
if (b.ENVIRONMENT) {
  throw Error("Module.ENVIRONMENT has been deprecated. To force the environment, use the ENVIRONMENT compile-time option (for example, -s ENVIRONMENT=web or -s ENVIRONMENT=node)");
}
var t = "", ka, la, ma, fs, na, oa;
if (ia) {
  if ("object" !== typeof process || "function" !== typeof require) {
    throw Error("not compiled for this environment (did you build to HTML and try to run it not on the web, or set ENVIRONMENT to something - like node - and run it someplace else - like on the web?)");
  }
  t = ha ? require("path").dirname(t) + "/" : __dirname + "/";
  oa = () => {
    na || (fs = require("fs"), na = require("path"));
  };
  ka = function(a, c) {
    oa();
    a = na.normalize(a);
    return fs.readFileSync(a, c ? null : "utf8");
  };
  ma = a => {
    a = ka(a, !0);
    a.buffer || (a = new Uint8Array(a));
    assert(a.buffer);
    return a;
  };
  la = (a, c, d) => {
    oa();
    a = na.normalize(a);
    fs.readFile(a, function(e, f) {
      e ? d(e) : c(f.buffer);
    });
  };
  1 < process.argv.length && (ea = process.argv[1].replace(/\\/g, "/"));
  process.argv.slice(2);
  process.on("uncaughtException", function(a) {
    throw a;
  });
  process.on("unhandledRejection", function(a) {
    throw a;
  });
  b.inspect = function() {
    return "[Emscripten Module object]";
  };
} else if (ja) {
  if ("object" === typeof process && "function" === typeof require || "object" === typeof window || "function" === typeof importScripts) {
    throw Error("not compiled for this environment (did you build to HTML and try to run it not on the web, or set ENVIRONMENT to something - like node - and run it someplace else - like on the web?)");
  }
  "undefined" != typeof read && (ka = function(a) {
    return read(a);
  });
  ma = function(a) {
    if ("function" === typeof readbuffer) {
      return new Uint8Array(readbuffer(a));
    }
    a = read(a, "binary");
    assert("object" === typeof a);
    return a;
  };
  la = function(a, c) {
    setTimeout(() => c(ma(a)), 0);
  };
  "undefined" !== typeof print && ("undefined" === typeof console && (console = {}), console.log = print, console.warn = console.error = "undefined" !== typeof printErr ? printErr : print);
} else if (fa || ha) {
  ha ? t = self.location.href : "undefined" !== typeof document && document.currentScript && (t = document.currentScript.src);
  _scriptDir && (t = _scriptDir);
  0 !== t.indexOf("blob:") ? t = t.substr(0, t.replace(/[?#].*/, "").lastIndexOf("/") + 1) : t = "";
  if ("object" !== typeof window && "function" !== typeof importScripts) {
    throw Error("not compiled for this environment (did you build to HTML and try to run it not on the web, or set ENVIRONMENT to something - like node - and run it someplace else - like on the web?)");
  }
  ka = a => {
    var c = new XMLHttpRequest();
    c.open("GET", a, !1);
    c.send(null);
    return c.responseText;
  };
  ha && (ma = a => {
    var c = new XMLHttpRequest();
    c.open("GET", a, !1);
    c.responseType = "arraybuffer";
    c.send(null);
    return new Uint8Array(c.response);
  });
  la = (a, c, d) => {
    var e = new XMLHttpRequest();
    e.open("GET", a, !0);
    e.responseType = "arraybuffer";
    e.onload = () => {
      200 == e.status || 0 == e.status && e.response ? c(e.response) : d();
    };
    e.onerror = d;
    e.send(null);
  };
} else {
  throw Error("environment detection error");
}
var pa = b.print || console.log.bind(console), u = b.printErr || console.warn.bind(console);
aa(b, da);
da = null;
Object.getOwnPropertyDescriptor(b, "arguments") || Object.defineProperty(b, "arguments", {configurable:!0, get:function() {
  n("Module.arguments has been replaced with plain arguments_ (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
b.thisProgram && (ea = b.thisProgram);
Object.getOwnPropertyDescriptor(b, "thisProgram") || Object.defineProperty(b, "thisProgram", {configurable:!0, get:function() {
  n("Module.thisProgram has been replaced with plain thisProgram (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
Object.getOwnPropertyDescriptor(b, "quit") || Object.defineProperty(b, "quit", {configurable:!0, get:function() {
  n("Module.quit has been replaced with plain quit_ (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
assert("undefined" === typeof b.memoryInitializerPrefixURL, "Module.memoryInitializerPrefixURL option was removed, use Module.locateFile instead");
assert("undefined" === typeof b.pthreadMainPrefixURL, "Module.pthreadMainPrefixURL option was removed, use Module.locateFile instead");
assert("undefined" === typeof b.cdInitializerPrefixURL, "Module.cdInitializerPrefixURL option was removed, use Module.locateFile instead");
assert("undefined" === typeof b.filePackagePrefixURL, "Module.filePackagePrefixURL option was removed, use Module.locateFile instead");
assert("undefined" === typeof b.read, "Module.read option was removed (modify read_ in JS)");
assert("undefined" === typeof b.readAsync, "Module.readAsync option was removed (modify readAsync in JS)");
assert("undefined" === typeof b.readBinary, "Module.readBinary option was removed (modify readBinary in JS)");
assert("undefined" === typeof b.setWindowTitle, "Module.setWindowTitle option was removed (modify setWindowTitle in JS)");
assert("undefined" === typeof b.TOTAL_MEMORY, "Module.TOTAL_MEMORY has been renamed Module.INITIAL_MEMORY");
Object.getOwnPropertyDescriptor(b, "read") || Object.defineProperty(b, "read", {configurable:!0, get:function() {
  n("Module.read has been replaced with plain read_ (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
Object.getOwnPropertyDescriptor(b, "readAsync") || Object.defineProperty(b, "readAsync", {configurable:!0, get:function() {
  n("Module.readAsync has been replaced with plain readAsync (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
Object.getOwnPropertyDescriptor(b, "readBinary") || Object.defineProperty(b, "readBinary", {configurable:!0, get:function() {
  n("Module.readBinary has been replaced with plain readBinary (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
Object.getOwnPropertyDescriptor(b, "setWindowTitle") || Object.defineProperty(b, "setWindowTitle", {configurable:!0, get:function() {
  n("Module.setWindowTitle has been replaced with plain setWindowTitle (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
assert(!ja, "shell environment detected but not enabled at build time.  Add 'shell' to `-s ENVIRONMENT` to enable.");
function qa(a) {
  ra || (ra = {});
  ra[a] || (ra[a] = 1, u(a));
}
var ra, v = 0, ta;
b.wasmBinary && (ta = b.wasmBinary);
Object.getOwnPropertyDescriptor(b, "wasmBinary") || Object.defineProperty(b, "wasmBinary", {configurable:!0, get:function() {
  n("Module.wasmBinary has been replaced with plain wasmBinary (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
var noExitRuntime = b.noExitRuntime || !1;
Object.getOwnPropertyDescriptor(b, "noExitRuntime") || Object.defineProperty(b, "noExitRuntime", {configurable:!0, get:function() {
  n("Module.noExitRuntime has been replaced with plain noExitRuntime (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
"object" !== typeof WebAssembly && n("no native wasm support detected");
function ua(a, c, d = "i8") {
  "*" === d.charAt(d.length - 1) && (d = "i32");
  switch(d) {
    case "i1":
      x[a >> 0] = c;
      break;
    case "i8":
      x[a >> 0] = c;
      break;
    case "i16":
      va[a >> 1] = c;
      break;
    case "i32":
      wa[a >> 2] = c;
      break;
    case "i64":
      xa = [c >>> 0, (ya = c, 1 <= +Math.abs(ya) ? 0 < ya ? (Math.min(+Math.floor(ya / 4294967296), 4294967295) | 0) >>> 0 : ~~+Math.ceil((ya - +(~~ya >>> 0)) / 4294967296) >>> 0 : 0)];
      wa[a >> 2] = xa[0];
      wa[a + 4 >> 2] = xa[1];
      break;
    case "float":
      za[a >> 2] = c;
      break;
    case "double":
      Aa[a >> 3] = c;
      break;
    default:
      n("invalid type for setValue: " + d);
  }
}
function Ba(a, c = "i8") {
  "*" === c.charAt(c.length - 1) && (c = "i32");
  switch(c) {
    case "i1":
      return x[a >> 0];
    case "i8":
      return x[a >> 0];
    case "i16":
      return va[a >> 1];
    case "i32":
      return wa[a >> 2];
    case "i64":
      return wa[a >> 2];
    case "float":
      return za[a >> 2];
    case "double":
      return Number(Aa[a >> 3]);
    default:
      n("invalid type for getValue: " + c);
  }
  return null;
}
function Ca(a) {
  switch(a) {
    case 1:
      return "i8";
    case 2:
      return "i16";
    case 4:
      return "i32";
    case 8:
      return "i64";
    default:
      assert(0);
  }
}
function B(a, c, d) {
  0 >= a && n("segmentation fault storing " + d + " bytes to address " + a);
  0 !== a % d && n("alignment error storing to address " + a + ", which was expected to be aligned to a multiple of " + d);
  if (Da && !Ea) {
    var e = Fa() >>> 0;
    a + d > e && n("segmentation fault, exceeded the top of the available dynamic heap when storing " + d + " bytes to address " + a + ". DYNAMICTOP=" + e);
    assert(e >= Ga());
    assert(e <= x.length);
  }
  ua(a, c, Ca(d));
}
function D(a, c, d) {
  0 >= a && n("segmentation fault loading " + c + " bytes from address " + a);
  0 !== a % c && n("alignment error loading from address " + a + ", which was expected to be aligned to a multiple of " + c);
  if (Da && !Ea) {
    var e = Fa() >>> 0;
    a + c > e && n("segmentation fault, exceeded the top of the available dynamic heap when loading " + c + " bytes from address " + a + ". DYNAMICTOP=" + e);
    assert(e >= Ga());
    assert(e <= x.length);
  }
  c = Ca(c);
  a = Ba(a, c);
  d && (d = parseInt(c.substr(1), 10), a = 0 <= a ? a : 32 >= d ? 2 * Math.abs(1 << d - 1) + a : Math.pow(2, d) + a);
  return a;
}
var Ha, Ia = !1;
function assert(a, c) {
  a || n("Assertion failed" + (c ? ": " + c : ""));
}
var Ja = "undefined" !== typeof TextDecoder ? new TextDecoder("utf8") : void 0;
function Ka(a, c, d) {
  var e = c + d;
  for (d = c; a[d] && !(d >= e);) {
    ++d;
  }
  if (16 < d - c && a.subarray && Ja) {
    return Ja.decode(a.subarray(c, d));
  }
  for (e = ""; c < d;) {
    var f = a[c++];
    if (f & 128) {
      var g = a[c++] & 63;
      if (192 == (f & 224)) {
        e += String.fromCharCode((f & 31) << 6 | g);
      } else {
        var h = a[c++] & 63;
        224 == (f & 240) ? f = (f & 15) << 12 | g << 6 | h : (240 != (f & 248) && qa("Invalid UTF-8 leading byte 0x" + f.toString(16) + " encountered when deserializing a UTF-8 string in wasm memory to a JS string!"), f = (f & 7) << 18 | g << 12 | h << 6 | a[c++] & 63);
        65536 > f ? e += String.fromCharCode(f) : (f -= 65536, e += String.fromCharCode(55296 | f >> 10, 56320 | f & 1023));
      }
    } else {
      e += String.fromCharCode(f);
    }
  }
  return e;
}
function F(a, c) {
  return a ? Ka(La, a, c) : "";
}
function Ma(a, c, d, e) {
  if (!(0 < e)) {
    return 0;
  }
  var f = d;
  e = d + e - 1;
  for (var g = 0; g < a.length; ++g) {
    var h = a.charCodeAt(g);
    if (55296 <= h && 57343 >= h) {
      var k = a.charCodeAt(++g);
      h = 65536 + ((h & 1023) << 10) | k & 1023;
    }
    if (127 >= h) {
      if (d >= e) {
        break;
      }
      c[d++] = h;
    } else {
      if (2047 >= h) {
        if (d + 1 >= e) {
          break;
        }
        c[d++] = 192 | h >> 6;
      } else {
        if (65535 >= h) {
          if (d + 2 >= e) {
            break;
          }
          c[d++] = 224 | h >> 12;
        } else {
          if (d + 3 >= e) {
            break;
          }
          1114111 < h && qa("Invalid Unicode code point 0x" + h.toString(16) + " encountered when serializing a JS string to a UTF-8 string in wasm memory! (Valid unicode code points should be in range 0-0x10FFFF).");
          c[d++] = 240 | h >> 18;
          c[d++] = 128 | h >> 12 & 63;
        }
        c[d++] = 128 | h >> 6 & 63;
      }
      c[d++] = 128 | h & 63;
    }
  }
  c[d] = 0;
  return d - f;
}
function Na(a) {
  for (var c = 0, d = 0; d < a.length; ++d) {
    var e = a.charCodeAt(d);
    55296 <= e && 57343 >= e && (e = 65536 + ((e & 1023) << 10) | a.charCodeAt(++d) & 1023);
    127 >= e ? ++c : c = 2047 >= e ? c + 2 : 65535 >= e ? c + 3 : c + 4;
  }
  return c;
}
"undefined" !== typeof TextDecoder && new TextDecoder("utf-16le");
function Oa(a) {
  var c = Na(a) + 1, d = Pa(c);
  d && Ma(a, x, d, c);
  return d;
}
function Qa(a, c) {
  assert(0 <= a.length, "writeArrayToMemory array must have a length (should be an array or typed array)");
  x.set(a, c);
}
var Ra, x, La, va, wa, za, Aa;
function Sa() {
  var a = Ha.buffer;
  Ra = a;
  b.HEAP8 = x = new Int8Array(a);
  b.HEAP16 = va = new Int16Array(a);
  b.HEAP32 = wa = new Int32Array(a);
  b.HEAPU8 = La = new Uint8Array(a);
  b.HEAPU16 = new Uint16Array(a);
  b.HEAPU32 = new Uint32Array(a);
  b.HEAPF32 = za = new Float32Array(a);
  b.HEAPF64 = Aa = new Float64Array(a);
}
b.TOTAL_STACK && assert(5242880 === b.TOTAL_STACK, "the stack size can no longer be determined at runtime");
var Ta = b.INITIAL_MEMORY || 16777216;
Object.getOwnPropertyDescriptor(b, "INITIAL_MEMORY") || Object.defineProperty(b, "INITIAL_MEMORY", {configurable:!0, get:function() {
  n("Module.INITIAL_MEMORY has been replaced with plain INITIAL_MEMORY (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
assert(5242880 <= Ta, "INITIAL_MEMORY should be larger than TOTAL_STACK, was " + Ta + "! (TOTAL_STACK=5242880)");
assert("undefined" !== typeof Int32Array && "undefined" !== typeof Float64Array && void 0 !== Int32Array.prototype.subarray && void 0 !== Int32Array.prototype.set, "JS engine does not provide full typed array support");
assert(!b.wasmMemory, "Use of `wasmMemory` detected.  Use -s IMPORTED_MEMORY to define wasmMemory externally");
assert(16777216 == Ta, "Detected runtime INITIAL_MEMORY setting.  Use -s IMPORTED_MEMORY to define wasmMemory dynamically");
var Ua;
function Va() {
  var a = Wa();
  assert(0 == (a & 3));
  B(a + 4 | 0, 34821223, 4);
  B(a + 8 | 0, -1984246274, 4);
}
function Xa() {
  if (!Ia) {
    var a = Wa(), c = D(a + 4 | 0, 4, 1) >>> 0;
    a = D(a + 8 | 0, 4, 1) >>> 0;
    34821223 == c && 2310721022 == a || n("Stack overflow! Stack cookie has been overwritten, expected hex dwords 0x89BACDFE and 0x2135467, but received 0x" + a.toString(16) + " 0x" + c.toString(16));
  }
}
var Ya = new Int16Array(1), Za = new Int8Array(Ya.buffer);
Ya[0] = 25459;
if (115 !== Za[0] || 99 !== Za[1]) {
  throw "Runtime error: expected the system to be little-endian! (Run with -s SUPPORT_BIG_ENDIAN=1 to bypass)";
}
var $a = [], ab = [], bb = [], Da = !1, Ea = !1;
function cb() {
  var a = b.preRun.shift();
  $a.unshift(a);
}
assert(Math.imul, "This browser does not support Math.imul(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill");
assert(Math.fround, "This browser does not support Math.fround(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill");
assert(Math.clz32, "This browser does not support Math.clz32(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill");
assert(Math.trunc, "This browser does not support Math.trunc(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill");
var db = 0, eb = null, fb = null, gb = {};
function hb() {
  db++;
  b.monitorRunDependencies && b.monitorRunDependencies(db);
  assert(!gb["wasm-instantiate"]);
  gb["wasm-instantiate"] = 1;
  null === eb && "undefined" !== typeof setInterval && (eb = setInterval(function() {
    if (Ia) {
      clearInterval(eb), eb = null;
    } else {
      var a = !1, c;
      for (c in gb) {
        a || (a = !0, u("still waiting on run dependencies:")), u("dependency: " + c);
      }
      a && u("(end of list)");
    }
  }, 1e4));
}
b.preloadedImages = {};
b.preloadedAudios = {};
function n(a) {
  if (b.onAbort) {
    b.onAbort(a);
  }
  a = "Aborted(" + a + ")";
  u(a);
  Ia = !0;
  a = new WebAssembly.RuntimeError(a);
  ca(a);
  throw a;
}
function ib() {
  n("Filesystem support (FS) was not included. The problem is that you are using files from JS, but files were not used from C/C++, so filesystem support was not auto-included. You can force-include filesystem support with  -s FORCE_FILESYSTEM=1");
}
b.FS_createDataFile = function() {
  ib();
};
b.FS_createPreloadedFile = function() {
  ib();
};
function jb() {
  return I.startsWith("data:application/octet-stream;base64,");
}
function L(a) {
  return function() {
    var c = b.asm;
    assert(Da, "native function `" + a + "` called before runtime initialization");
    assert(!Ea, "native function `" + a + "` called after runtime exit (use NO_EXIT_RUNTIME to keep it alive after main() exits)");
    c[a] || assert(c[a], "exported native function `" + a + "` not found");
    return c[a].apply(null, arguments);
  };
}
var I;
I = "ort-wasm.wasm";
if (!jb()) {
  var kb = I;
  I = b.locateFile ? b.locateFile(kb, t) : t + kb;
}
function lb() {
  var a = I;
  try {
    if (a == I && ta) {
      return new Uint8Array(ta);
    }
    if (ma) {
      return ma(a);
    }
    throw "both async and sync fetching of the wasm failed";
  } catch (c) {
    n(c);
  }
}
function mb() {
  if (!ta && (fa || ha)) {
    if ("function" === typeof fetch && !I.startsWith("file://")) {
      return fetch(I, {credentials:"same-origin"}).then(function(a) {
        if (!a.ok) {
          throw "failed to load wasm binary file at '" + I + "'";
        }
        return a.arrayBuffer();
      }).catch(function() {
        return lb();
      });
    }
    if (la) {
      return new Promise(function(a, c) {
        la(I, function(d) {
          a(new Uint8Array(d));
        }, c);
      });
    }
  }
  return Promise.resolve().then(function() {
    return lb();
  });
}
var ya, xa;
function nb(a) {
  for (; 0 < a.length;) {
    var c = a.shift();
    if ("function" == typeof c) {
      c(b);
    } else {
      var d = c.ga;
      "number" === typeof d ? void 0 === c.D ? N(d)() : N(d)(c.D) : d(void 0 === c.D ? null : c.D);
    }
  }
}
var ob = [];
function N(a) {
  var c = ob[a];
  c || (a >= ob.length && (ob.length = a + 1), ob[a] = c = Ua.get(a));
  assert(Ua.get(a) == c, "JavaScript-side Wasm function table mirror is out of date!");
  return c;
}
function pb(a) {
  this.I = a;
  this.g = a - 16;
  this.Y = function(c) {
    B(this.g + 4 | 0, c | 0, 4);
  };
  this.l = function() {
    return D(this.g + 4 | 0, 4, 0) | 0;
  };
  this.W = function(c) {
    B(this.g + 8 | 0, c | 0, 4);
  };
  this.P = function() {
    return D(this.g + 8 | 0, 4, 0) | 0;
  };
  this.X = function() {
    B(this.g | 0, 0, 4);
  };
  this.G = function(c) {
    B(this.g + 12 | 0, (c ? 1 : 0) | 0, 1);
  };
  this.O = function() {
    return 0 != (D(this.g + 12 | 0, 1, 0) | 0);
  };
  this.H = function(c) {
    B(this.g + 13 | 0, (c ? 1 : 0) | 0, 1);
  };
  this.J = function() {
    return 0 != (D(this.g + 13 | 0, 1, 0) | 0);
  };
  this.S = function(c, d) {
    this.Y(c);
    this.W(d);
    this.X();
    this.G(!1);
    this.H(!1);
  };
  this.K = function() {
    var c = D(this.g | 0, 4, 0) | 0;
    B(this.g | 0, c + 1 | 0, 4);
  };
  this.V = function() {
    var c = D(this.g | 0, 4, 0) | 0;
    B(this.g | 0, c - 1 | 0, 4);
    assert(0 < c);
    return 1 === c;
  };
}
function qb(a) {
  this.F = function() {
    rb(this.g);
    this.g = 0;
  };
  this.v = function(c) {
    B(this.g | 0, c | 0, 4);
  };
  this.j = function() {
    return D(this.g | 0, 4, 0) | 0;
  };
  this.m = function(c) {
    B(this.g + 4 | 0, c | 0, 4);
  };
  this.s = function() {
    return this.g + 4;
  };
  this.N = function() {
    return D(this.g + 4 | 0, 4, 0) | 0;
  };
  this.R = function() {
    if (sb(this.u().l())) {
      return D(this.j() | 0, 4, 0) | 0;
    }
    var c = this.N();
    return 0 !== c ? c : this.j();
  };
  this.u = function() {
    return new pb(this.j());
  };
  void 0 === a ? (this.g = Pa(8), this.m(0)) : this.g = a;
}
var tb = [], ub = 0, Q = 0;
function vb(a) {
  try {
    return rb((new pb(a)).g);
  } catch (c) {
    u("exception during cxa_free_exception: " + c);
  }
}
var wb = {}, xb = [null, [], []], yb = {};
function zb(a, c, d) {
  function e(l) {
    return (l = l.toTimeString().match(/\(([A-Za-z ]+)\)$/)) ? l[1] : "GMT";
  }
  var f = (new Date()).getFullYear(), g = new Date(f, 0, 1), h = new Date(f, 6, 1);
  f = g.getTimezoneOffset();
  var k = h.getTimezoneOffset();
  B(a | 0, 60 * Math.max(f, k) | 0, 4);
  B(c | 0, Number(f != k) | 0, 4);
  a = e(g);
  c = e(h);
  a = Oa(a);
  c = Oa(c);
  k < f ? (B(d | 0, a | 0, 4), B(d + 4 | 0, c | 0, 4)) : (B(d | 0, c | 0, 4), B(d + 4 | 0, a | 0, 4));
}
function Ab(a, c, d) {
  Ab.M || (Ab.M = !0, zb(a, c, d));
}
var Bb;
Bb = ia ? () => {
  var a = process.hrtime();
  return 1e3 * a[0] + a[1] / 1e6;
} : () => performance.now();
var Cb = {};
function Db() {
  if (!Eb) {
    var a = {USER:"web_user", LOGNAME:"web_user", PATH:"/", PWD:"/", HOME:"/home/web_user", LANG:("object" === typeof navigator && navigator.languages && navigator.languages[0] || "C").replace("-", "_") + ".UTF-8", _:ea || "./this.program"}, c;
    for (c in Cb) {
      void 0 === Cb[c] ? delete a[c] : a[c] = Cb[c];
    }
    var d = [];
    for (c in a) {
      d.push(c + "=" + a[c]);
    }
    Eb = d;
  }
  return Eb;
}
var Eb;
function Fb(a) {
  return 0 === a % 4 && (0 !== a % 100 || 0 === a % 400);
}
function Gb(a, c) {
  for (var d = 0, e = 0; e <= c; d += a[e++]) {
  }
  return d;
}
var Hb = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], Ib = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
function Jb(a, c) {
  for (a = new Date(a.getTime()); 0 < c;) {
    var d = a.getMonth(), e = (Fb(a.getFullYear()) ? Hb : Ib)[d];
    if (c > e - a.getDate()) {
      c -= e - a.getDate() + 1, a.setDate(1), 11 > d ? a.setMonth(d + 1) : (a.setMonth(0), a.setFullYear(a.getFullYear() + 1));
    } else {
      a.setDate(a.getDate() + c);
      break;
    }
  }
  return a;
}
function Kb(a, c, d, e) {
  function f(p, w, z) {
    for (p = "number" === typeof p ? p.toString() : p || ""; p.length < w;) {
      p = z[0] + p;
    }
    return p;
  }
  function g(p, w) {
    return f(p, w, "0");
  }
  function h(p, w) {
    function z(C) {
      return 0 > C ? -1 : 0 < C ? 1 : 0;
    }
    var A;
    0 === (A = z(p.getFullYear() - w.getFullYear())) && 0 === (A = z(p.getMonth() - w.getMonth())) && (A = z(p.getDate() - w.getDate()));
    return A;
  }
  function k(p) {
    switch(p.getDay()) {
      case 0:
        return new Date(p.getFullYear() - 1, 11, 29);
      case 1:
        return p;
      case 2:
        return new Date(p.getFullYear(), 0, 3);
      case 3:
        return new Date(p.getFullYear(), 0, 2);
      case 4:
        return new Date(p.getFullYear(), 0, 1);
      case 5:
        return new Date(p.getFullYear() - 1, 11, 31);
      case 6:
        return new Date(p.getFullYear() - 1, 11, 30);
    }
  }
  function l(p) {
    p = Jb(new Date(p.h + 1900, 0, 1), p.C);
    var w = new Date(p.getFullYear() + 1, 0, 4), z = k(new Date(p.getFullYear(), 0, 4));
    w = k(w);
    return 0 >= h(z, p) ? 0 >= h(w, p) ? p.getFullYear() + 1 : p.getFullYear() : p.getFullYear() - 1;
  }
  var m = D(e + 40 | 0, 4, 0) | 0;
  e = {aa:D(e | 0, 4, 0) | 0, $:D(e + 4 | 0, 4, 0) | 0, A:D(e + 8 | 0, 4, 0) | 0, o:D(e + 12 | 0, 4, 0) | 0, i:D(e + 16 | 0, 4, 0) | 0, h:D(e + 20 | 0, 4, 0) | 0, B:D(e + 24 | 0, 4, 0) | 0, C:D(e + 28 | 0, 4, 0) | 0, ja:D(e + 32 | 0, 4, 0) | 0, Z:D(e + 36 | 0, 4, 0) | 0, ba:m ? F(m) : ""};
  d = F(d);
  m = {"%c":"%a %b %d %H:%M:%S %Y", "%D":"%m/%d/%y", "%F":"%Y-%m-%d", "%h":"%b", "%r":"%I:%M:%S %p", "%R":"%H:%M", "%T":"%H:%M:%S", "%x":"%m/%d/%y", "%X":"%H:%M:%S", "%Ec":"%c", "%EC":"%C", "%Ex":"%m/%d/%y", "%EX":"%H:%M:%S", "%Ey":"%y", "%EY":"%Y", "%Od":"%d", "%Oe":"%e", "%OH":"%H", "%OI":"%I", "%Om":"%m", "%OM":"%M", "%OS":"%S", "%Ou":"%u", "%OU":"%U", "%OV":"%V", "%Ow":"%w", "%OW":"%W", "%Oy":"%y"};
  for (var q in m) {
    d = d.replace(new RegExp(q, "g"), m[q]);
  }
  var r = "Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "), y = "January February March April May June July August September October November December".split(" ");
  m = {"%a":function(p) {
    return r[p.B].substring(0, 3);
  }, "%A":function(p) {
    return r[p.B];
  }, "%b":function(p) {
    return y[p.i].substring(0, 3);
  }, "%B":function(p) {
    return y[p.i];
  }, "%C":function(p) {
    return g((p.h + 1900) / 100 | 0, 2);
  }, "%d":function(p) {
    return g(p.o, 2);
  }, "%e":function(p) {
    return f(p.o, 2, " ");
  }, "%g":function(p) {
    return l(p).toString().substring(2);
  }, "%G":function(p) {
    return l(p);
  }, "%H":function(p) {
    return g(p.A, 2);
  }, "%I":function(p) {
    p = p.A;
    0 == p ? p = 12 : 12 < p && (p -= 12);
    return g(p, 2);
  }, "%j":function(p) {
    return g(p.o + Gb(Fb(p.h + 1900) ? Hb : Ib, p.i - 1), 3);
  }, "%m":function(p) {
    return g(p.i + 1, 2);
  }, "%M":function(p) {
    return g(p.$, 2);
  }, "%n":function() {
    return "\n";
  }, "%p":function(p) {
    return 0 <= p.A && 12 > p.A ? "AM" : "PM";
  }, "%S":function(p) {
    return g(p.aa, 2);
  }, "%t":function() {
    return "\t";
  }, "%u":function(p) {
    return p.B || 7;
  }, "%U":function(p) {
    var w = new Date(p.h + 1900, 0, 1), z = 0 === w.getDay() ? w : Jb(w, 7 - w.getDay());
    p = new Date(p.h + 1900, p.i, p.o);
    return 0 > h(z, p) ? g(Math.ceil((31 - z.getDate() + (Gb(Fb(p.getFullYear()) ? Hb : Ib, p.getMonth() - 1) - 31) + p.getDate()) / 7), 2) : 0 === h(z, w) ? "01" : "00";
  }, "%V":function(p) {
    var w = new Date(p.h + 1901, 0, 4), z = k(new Date(p.h + 1900, 0, 4));
    w = k(w);
    var A = Jb(new Date(p.h + 1900, 0, 1), p.C);
    return 0 > h(A, z) ? "53" : 0 >= h(w, A) ? "01" : g(Math.ceil((z.getFullYear() < p.h + 1900 ? p.C + 32 - z.getDate() : p.C + 1 - z.getDate()) / 7), 2);
  }, "%w":function(p) {
    return p.B;
  }, "%W":function(p) {
    var w = new Date(p.h, 0, 1), z = 1 === w.getDay() ? w : Jb(w, 0 === w.getDay() ? 1 : 7 - w.getDay() + 1);
    p = new Date(p.h + 1900, p.i, p.o);
    return 0 > h(z, p) ? g(Math.ceil((31 - z.getDate() + (Gb(Fb(p.getFullYear()) ? Hb : Ib, p.getMonth() - 1) - 31) + p.getDate()) / 7), 2) : 0 === h(z, w) ? "01" : "00";
  }, "%y":function(p) {
    return (p.h + 1900).toString().substring(2);
  }, "%Y":function(p) {
    return p.h + 1900;
  }, "%z":function(p) {
    p = p.Z;
    var w = 0 <= p;
    p = Math.abs(p) / 60;
    return (w ? "+" : "-") + String("0000" + (p / 60 * 100 + p % 60)).slice(-4);
  }, "%Z":function(p) {
    return p.ba;
  }, "%%":function() {
    return "%";
  }};
  for (q in m) {
    d.includes(q) && (d = d.replace(new RegExp(q, "g"), m[q](e)));
  }
  q = Lb(d);
  if (q.length > c) {
    return 0;
  }
  Qa(q, a);
  return q.length - 1;
}
function Lb(a) {
  var c = Array(Na(a) + 1);
  Ma(a, c, 0, c.length);
  return c;
}
var Mf = {__assert_fail:function(a, c, d, e) {
  n("Assertion failed: " + F(a) + ", at: " + [c ? F(c) : "unknown filename", d, e ? F(e) : "unknown function"]);
}, __cxa_allocate_exception:function(a) {
  return Pa(a + 16) + 16;
}, __cxa_begin_catch:function(a) {
  a = new qb(a);
  var c = a.u();
  c.O() || (c.G(!0), ub--);
  c.H(!1);
  tb.push(a);
  c.K();
  return a.R();
}, __cxa_call_unexpected:function(a) {
  u("Unexpected exception thrown, this is not properly supported - aborting");
  Ia = !0;
  throw a;
}, __cxa_end_catch:function() {
  V(0);
  assert(0 < tb.length);
  var a = tb.pop(), c = a.u();
  if (c.V() && !c.J()) {
    var d = c.P();
    d && N(d)(c.I);
    vb(c.I);
  }
  a.F();
  Q = 0;
}, __cxa_find_matching_catch_2:function() {
  var a = Q;
  if (!a) {
    return v = 0;
  }
  var c = (new pb(a)).l(), d = new qb();
  d.v(a);
  d.m(a);
  if (!c) {
    return v = 0, d.g | 0;
  }
  a = Array.prototype.slice.call(arguments);
  for (var e = 0; e < a.length; e++) {
    var f = a[e];
    if (0 === f || f === c) {
      break;
    }
    if (Mb(f, c, d.s())) {
      return v = f, d.g | 0;
    }
  }
  v = c;
  return d.g | 0;
}, __cxa_find_matching_catch_3:function() {
  var a = Q;
  if (!a) {
    return v = 0;
  }
  var c = (new pb(a)).l(), d = new qb();
  d.v(a);
  d.m(a);
  if (!c) {
    return v = 0, d.g | 0;
  }
  a = Array.prototype.slice.call(arguments);
  for (var e = 0; e < a.length; e++) {
    var f = a[e];
    if (0 === f || f === c) {
      break;
    }
    if (Mb(f, c, d.s())) {
      return v = f, d.g | 0;
    }
  }
  v = c;
  return d.g | 0;
}, __cxa_find_matching_catch_4:function() {
  var a = Q;
  if (!a) {
    return v = 0;
  }
  var c = (new pb(a)).l(), d = new qb();
  d.v(a);
  d.m(a);
  if (!c) {
    return v = 0, d.g | 0;
  }
  a = Array.prototype.slice.call(arguments);
  for (var e = 0; e < a.length; e++) {
    var f = a[e];
    if (0 === f || f === c) {
      break;
    }
    if (Mb(f, c, d.s())) {
      return v = f, d.g | 0;
    }
  }
  v = c;
  return d.g | 0;
}, __cxa_find_matching_catch_5:function() {
  var a = Q;
  if (!a) {
    return v = 0;
  }
  var c = (new pb(a)).l(), d = new qb();
  d.v(a);
  d.m(a);
  if (!c) {
    return v = 0, d.g | 0;
  }
  a = Array.prototype.slice.call(arguments);
  for (var e = 0; e < a.length; e++) {
    var f = a[e];
    if (0 === f || f === c) {
      break;
    }
    if (Mb(f, c, d.s())) {
      return v = f, d.g | 0;
    }
  }
  v = c;
  return d.g | 0;
}, __cxa_free_exception:vb, __cxa_rethrow:function() {
  var a = tb.pop();
  a || n("no exception to throw");
  var c = a.u(), d = a.j();
  c.J() ? a.F() : (tb.push(a), c.H(!0), c.G(!1), ub++);
  Q = d;
  throw d;
}, __cxa_throw:function(a, c, d) {
  (new pb(a)).S(c, d);
  Q = a;
  ub++;
  throw a;
}, __cxa_uncaught_exceptions:function() {
  return ub;
}, __resumeException:function(a) {
  a = new qb(a);
  var c = a.j();
  Q || (Q = c);
  a.F();
  throw c;
}, __syscall_fcntl64:function() {
  return 0;
}, __syscall_fstat64:function() {
  n("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}, __syscall_fstatat64:function() {
  n("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}, __syscall_getcwd:function() {
  n("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}, __syscall_getdents64:function() {
  n("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}, __syscall_ioctl:function() {
  return 0;
}, __syscall_lstat64:function() {
  n("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}, __syscall_mkdir:function(a, c) {
  a = F(a);
  return yb.da(a, c);
}, __syscall_mmap2:function(a, c, d, e, f, g) {
  g <<= 12;
  if (0 !== (e & 16) && 0 !== a % 65536) {
    c = -28;
  } else {
    if (0 !== (e & 32)) {
      a = c;
      assert(65536, "alignment argument is required");
      a = 65536 * Math.ceil(a / 65536);
      var h = Nb(65536, a);
      h ? (La.fill(0, h, h + a), a = h) : a = 0;
      a ? (wb[a] = {U:a, T:c, L:!0, fd:f, ia:d, flags:e, offset:g}, c = a) : c = -48;
    } else {
      c = -52;
    }
  }
  return c;
}, __syscall_munmap:function(a, c) {
  var d = wb[a];
  0 !== c && d ? (c === d.T && (assert(wb[a].flags & 32), wb[a] = null, d.L && rb(d.U)), a = 0) : a = -28;
  return a;
}, __syscall_open:function() {
  n("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}, __syscall_readlink:function(a, c, d) {
  a = F(a);
  return yb.ea(a, c, d);
}, __syscall_rmdir:function() {
  n("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}, __syscall_stat64:function() {
  n("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}, __syscall_unlink:function() {
  n("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}, _dlopen_js:function() {
  n("To use dlopen, you need to use Emscripten's linking support, see https://github.com/emscripten-core/emscripten/wiki/Linking");
}, _dlsym_js:function() {
  n("To use dlopen, you need to use Emscripten's linking support, see https://github.com/emscripten-core/emscripten/wiki/Linking");
}, _gmtime_js:function(a, c) {
  a = new Date(1e3 * (D(a | 0, 4, 0) | 0));
  B(c | 0, a.getUTCSeconds() | 0, 4);
  B(c + 4 | 0, a.getUTCMinutes() | 0, 4);
  B(c + 8 | 0, a.getUTCHours() | 0, 4);
  B(c + 12 | 0, a.getUTCDate() | 0, 4);
  B(c + 16 | 0, a.getUTCMonth() | 0, 4);
  B(c + 20 | 0, a.getUTCFullYear() - 1900 | 0, 4);
  B(c + 24 | 0, a.getUTCDay() | 0, 4);
  B(c + 28 | 0, (a.getTime() - Date.UTC(a.getUTCFullYear(), 0, 1, 0, 0, 0, 0)) / 864E5 | 0, 4);
}, _localtime_js:function(a, c) {
  a = new Date(1e3 * (D(a | 0, 4, 0) | 0));
  B(c | 0, a.getSeconds() | 0, 4);
  B(c + 4 | 0, a.getMinutes() | 0, 4);
  B(c + 8 | 0, a.getHours() | 0, 4);
  B(c + 12 | 0, a.getDate() | 0, 4);
  B(c + 16 | 0, a.getMonth() | 0, 4);
  B(c + 20 | 0, a.getFullYear() - 1900 | 0, 4);
  B(c + 24 | 0, a.getDay() | 0, 4);
  var d = new Date(a.getFullYear(), 0, 1);
  B(c + 28 | 0, (a.getTime() - d.getTime()) / 864E5 | 0, 4);
  B(c + 36 | 0, -(60 * a.getTimezoneOffset()) | 0, 4);
  var e = (new Date(a.getFullYear(), 6, 1)).getTimezoneOffset();
  d = d.getTimezoneOffset();
  B(c + 32 | 0, (e != d && a.getTimezoneOffset() == Math.min(d, e)) | 0, 4);
}, _mktime_js:function(a) {
  var c = new Date((D(a + 20 | 0, 4, 0) | 0) + 1900, D(a + 16 | 0, 4, 0) | 0, D(a + 12 | 0, 4, 0) | 0, D(a + 8 | 0, 4, 0) | 0, D(a + 4 | 0, 4, 0) | 0, D(a | 0, 4, 0) | 0, 0), d = D(a + 32 | 0, 4, 0) | 0, e = c.getTimezoneOffset(), f = new Date(c.getFullYear(), 0, 1), g = (new Date(c.getFullYear(), 6, 1)).getTimezoneOffset(), h = f.getTimezoneOffset(), k = Math.min(h, g);
  0 > d ? B(a + 32 | 0, Number(g != h && k == e) | 0, 4) : 0 < d != (k == e) && (g = Math.max(h, g), c.setTime(c.getTime() + 6e4 * ((0 < d ? k : g) - e)));
  B(a + 24 | 0, c.getDay() | 0, 4);
  B(a + 28 | 0, (c.getTime() - f.getTime()) / 864E5 | 0, 4);
  B(a | 0, c.getSeconds() | 0, 4);
  B(a + 4 | 0, c.getMinutes() | 0, 4);
  B(a + 8 | 0, c.getHours() | 0, 4);
  B(a + 12 | 0, c.getDate() | 0, 4);
  B(a + 16 | 0, c.getMonth() | 0, 4);
  return c.getTime() / 1e3 | 0;
}, _tzset_js:Ab, abort:function() {
  n("native code called abort()");
}, alignfault:function() {
  n("alignment fault");
}, clock_gettime:function(a, c) {
  if (0 === a) {
    a = Date.now();
  } else if (1 === a || 4 === a) {
    a = Bb();
  } else {
    return B(Ob() | 0, 28, 4), -1;
  }
  B(c | 0, a / 1e3 | 0, 4);
  B(c + 4 | 0, a % 1e3 * 1E6 | 0, 4);
  return 0;
}, difftime:function(a, c) {
  return a - c;
}, emscripten_get_heap_max:function() {
  return 2147483648;
}, emscripten_get_now:Bb, emscripten_memcpy_big:function(a, c, d) {
  La.copyWithin(a, c, c + d);
}, emscripten_resize_heap:function(a) {
  var c = La.length;
  a >>>= 0;
  assert(a > c);
  if (2147483648 < a) {
    return u("Cannot enlarge memory, asked to go up to " + a + " bytes, but the limit is 2147483648 bytes!"), !1;
  }
  for (var d = 1; 4 >= d; d *= 2) {
    var e = c * (1 + .2 / d);
    e = Math.min(e, a + 100663296);
    e = Math.max(a, e);
    0 < e % 65536 && (e += 65536 - e % 65536);
    e = Math.min(2147483648, e);
    var f = Bb();
    a: {
      var g = e;
      try {
        Ha.grow(g - Ra.byteLength + 65535 >>> 16);
        Sa();
        var h = 1;
        break a;
      } catch (k) {
        u("emscripten_realloc_buffer: Attempted to grow heap from " + Ra.byteLength + " bytes to " + g + " bytes, but got error: " + k);
      }
      h = void 0;
    }
    g = Bb();
    pa("Heap resize call from " + c + " to " + e + " took " + (g - f) + " msecs. Success: " + !!h);
    if (h) {
      return !0;
    }
  }
  u("Failed to grow the heap from " + c + " bytes to " + e + " bytes, not enough memory!");
  return !1;
}, environ_get:function(a, c) {
  var d = 0;
  Db().forEach(function(e, f) {
    var g = c + d;
    B(a + 4 * f | 0, g | 0, 4);
    f = g;
    for (g = 0; g < e.length; ++g) {
      assert(e.charCodeAt(g) === (e.charCodeAt(g) & 255)), B(f++ | 0, e.charCodeAt(g) | 0, 1);
    }
    B(f | 0, 0, 1);
    d += e.length + 1;
  });
  return 0;
}, environ_sizes_get:function(a, c) {
  var d = Db();
  B(a | 0, d.length | 0, 4);
  var e = 0;
  d.forEach(function(f) {
    e += f.length + 1;
  });
  B(c | 0, e | 0, 4);
  return 0;
}, fd_close:function() {
  n("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
  return 0;
}, fd_read:function(a, c, d, e) {
  a = yb.ha(a);
  c = yb.fa(a, c, d);
  B(e | 0, c | 0, 4);
  return 0;
}, fd_seek:function() {
  n("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}, fd_write:function(a, c, d, e) {
  for (var f = 0, g = 0; g < d; g++) {
    var h = D(c | 0, 4, 0) | 0, k = D(c + 4 | 0, 4, 0) | 0;
    c += 8;
    for (var l = 0; l < k; l++) {
      var m = a, q = D(h + l, 1, 1), r = xb[m];
      assert(r);
      0 === q || 10 === q ? ((1 === m ? pa : u)(Ka(r, 0)), r.length = 0) : r.push(q);
    }
    f += k;
  }
  B(e | 0, f | 0, 4);
  return 0;
}, getTempRet0:function() {
  return v;
}, gettimeofday:function(a) {
  var c = Date.now();
  B(a | 0, c / 1e3 | 0, 4);
  B(a + 4 | 0, c % 1e3 * 1e3 | 0, 4);
  return 0;
}, invoke_di:Pb, invoke_dii:Qb, invoke_diii:Rb, invoke_fffffff:Sb, invoke_fi:Tb, invoke_fii:Ub, invoke_fiii:Vb, invoke_fijjjjifi:Wb, invoke_i:Xb, invoke_if:Yb, invoke_iffi:Zb, invoke_ii:$b, invoke_iidd:ac, invoke_iidi:bc, invoke_iif:cc, invoke_iiff:dc, invoke_iii:ec, invoke_iiifi:fc, invoke_iiifii:gc, invoke_iiii:hc, invoke_iiiii:ic, invoke_iiiiid:jc, invoke_iiiiidfffiii:kc, invoke_iiiiifiiiiii:lc, invoke_iiiiii:mc, invoke_iiiiiii:nc, invoke_iiiiiiii:oc, invoke_iiiiiiiii:pc, invoke_iiiiiiiiii:qc, 
invoke_iiiiiiiiiii:rc, invoke_iiiiiiiiiiii:sc, invoke_iiiiiiiiiiiii:tc, invoke_iiiiiiiiiiiiifi:uc, invoke_iiiiiiiiiiiiiiii:vc, invoke_iiiiiiiiiiiiiiiiifi:wc, invoke_iiiiiiiiiiiiiiiiiifi:xc, invoke_iiiiiiiiiji:yc, invoke_iiiiiiiiijii:zc, invoke_iiiiiiiiijj:Ac, invoke_iiiiiiiijjji:Bc, invoke_iiiiiij:Cc, invoke_iiiiiijji:Dc, invoke_iiiiiijjjii:Ec, invoke_iiiiij:Fc, invoke_iiiiiji:Gc, invoke_iiiiijiii:Hc, invoke_iiiiijiiiii:Ic, invoke_iiiiijji:Jc, invoke_iiiij:Kc, invoke_iiiiji:Lc, invoke_iiiijii:Mc, 
invoke_iiiijjii:Nc, invoke_iiiijjj:Oc, invoke_iiij:Pc, invoke_iiiji:Qc, invoke_iiijii:Rc, invoke_iiijiii:Sc, invoke_iiijiiiii:Tc, invoke_iiijj:Uc, invoke_iiijjii:Vc, invoke_iij:Wc, invoke_iiji:Xc, invoke_iijiiii:Yc, invoke_iijj:Zc, invoke_iijjjf:$c, invoke_iijjjii:ad, invoke_ij:bd, invoke_iji:cd, invoke_ijii:dd, invoke_ijiiii:ed, invoke_ijjj:fd, invoke_j:gd, invoke_jfi:hd, invoke_ji:jd, invoke_jii:kd, invoke_jiii:ld, invoke_jiiii:md, invoke_jiij:nd, invoke_jiji:od, invoke_jj:pd, invoke_jjj:qd, invoke_v:rd, 
invoke_vfiii:sd, invoke_vi:td, invoke_vid:ud, invoke_vidi:vd, invoke_vif:wd, invoke_viffiii:xd, invoke_vifi:yd, invoke_vifii:zd, invoke_vifiifiiii:Ad, invoke_vifiifiiiiiii:Bd, invoke_vifiii:Cd, invoke_vii:Dd, invoke_viid:Ed, invoke_viidi:Fd, invoke_viif:Gd, invoke_viiff:Hd, invoke_viifi:Id, invoke_viifiifijjjii:Jd, invoke_viifjjijiiii:Kd, invoke_viifjjjijiiiii:Ld, invoke_viii:Md, invoke_viiif:Nd, invoke_viiiff:Od, invoke_viiifii:Pd, invoke_viiifiifii:Qd, invoke_viiifiiii:Rd, invoke_viiifiiiii:Sd, 
invoke_viiifiiiiiifiiii:Td, invoke_viiii:Ud, invoke_viiiiff:Vd, invoke_viiiifiiifiii:Wd, invoke_viiiii:Xd, invoke_viiiiidiidii:Yd, invoke_viiiiif:Zd, invoke_viiiiiff:$d, invoke_viiiiifiifii:ae, invoke_viiiiifiiiifiii:be, invoke_viiiiifiiiiii:ce, invoke_viiiiii:de, invoke_viiiiiid:ee, invoke_viiiiiif:fe, invoke_viiiiiiff:ge, invoke_viiiiiii:he, invoke_viiiiiiidiiii:ie, invoke_viiiiiiifiiii:je, invoke_viiiiiiii:ke, invoke_viiiiiiiii:le, invoke_viiiiiiiiii:me, invoke_viiiiiiiiiii:ne, invoke_viiiiiiiiiiii:oe, 
invoke_viiiiiiiiiiiii:pe, invoke_viiiiiiiiiiiiii:qe, invoke_viiiiiiiiiiiiiifi:re, invoke_viiiiiiiiiiiiiii:se, invoke_viiiiiiiiiiiiiiii:te, invoke_viiiiiiiiiiiiiiiiiiiiiiiiii:ue, invoke_viiiiiiiijjj:ve, invoke_viiiiiiij:we, invoke_viiiiiiijiiii:xe, invoke_viiiiiiijiiiiiiiiiiiiiiiii:ye, invoke_viiiiiij:ze, invoke_viiiiiijjjjjii:Ae, invoke_viiiiij:Be, invoke_viiiiiji:Ce, invoke_viiiiijiiiiii:De, invoke_viiiiijjiiiii:Ee, invoke_viiiij:Fe, invoke_viiiiji:Ge, invoke_viiiijii:He, invoke_viiiijiiiiiiif:Ie, 
invoke_viiiijiiiiiiii:Je, invoke_viiiijj:Ke, invoke_viiiijjji:Le, invoke_viiij:Me, invoke_viiiji:Ne, invoke_viiijii:Oe, invoke_viiijiii:Pe, invoke_viiijiiiiiiiii:Qe, invoke_viiijiijjj:Re, invoke_viiijj:Se, invoke_viiijjiiiiiii:Te, invoke_viiijjjfffi:Ue, invoke_viiijjjfii:Ve, invoke_viiijjjii:We, invoke_viij:Xe, invoke_viiji:Ye, invoke_viijii:Ze, invoke_viijiiii:$e, invoke_viijiiiiiiiii:af, invoke_viijiiiiiiijjii:bf, invoke_viijiiiijii:cf, invoke_viijj:df, invoke_viijjiii:ef, invoke_viijjiiiiiiiii:ff, 
invoke_viijjj:gf, invoke_viijjjfiifiiii:hf, invoke_viijjjiiiii:jf, invoke_viijjjiiiiiii:kf, invoke_viijjjj:lf, invoke_viijjjjjjjjjjjjjif:mf, invoke_viijjjjjjjjjjjjjii:nf, invoke_vij:of, invoke_vijfjiiiii:pf, invoke_viji:qf, invoke_vijii:rf, invoke_vijiii:sf, invoke_vijiiiiii:tf, invoke_vijiiiiiiii:uf, invoke_vijiji:vf, invoke_vijjfffiii:wf, invoke_vijjiiii:xf, invoke_vijjj:yf, invoke_vijjjiij:zf, invoke_vijjjiiji:Af, invoke_vijjjjiii:Bf, invoke_vijjjjjjjjjjjjjii:Cf, invoke_vj:Df, invoke_vjifiii:Ef, 
invoke_vjiiiii:Ff, invoke_vjiiiiii:Gf, invoke_vjiiiiiii:Hf, invoke_vjjfiii:If, invoke_vjji:Jf, invoke_vjjjjjjffiifiiiii:Kf, invoke_vjjjjjjjjfffiifiiiii:Lf, llvm_eh_typeid_for:function(a) {
  return a;
}, segfault:function() {
  n("segmentation fault");
}, setTempRet0:function(a) {
  v = a;
}, strftime:Kb, strftime_l:function(a, c, d, e) {
  return Kb(a, c, d, e);
}};
(function() {
  function a(g) {
    b.asm = g.exports;
    Ha = b.asm.memory;
    assert(Ha, "memory not found in wasm exports");
    Sa();
    Ua = b.asm.__indirect_function_table;
    assert(Ua, "table not found in wasm exports");
    ab.unshift(b.asm.__wasm_call_ctors);
    db--;
    b.monitorRunDependencies && b.monitorRunDependencies(db);
    assert(gb["wasm-instantiate"]);
    delete gb["wasm-instantiate"];
    0 == db && (null !== eb && (clearInterval(eb), eb = null), fb && (g = fb, fb = null, g()));
  }
  function c(g) {
    assert(b === f, "the Module object should not be replaced during async compilation - perhaps the order of HTML elements is wrong?");
    f = null;
    a(g.instance);
  }
  function d(g) {
    return mb().then(function(h) {
      return WebAssembly.instantiate(h, e);
    }).then(function(h) {
      return h;
    }).then(g, function(h) {
      u("failed to asynchronously prepare wasm: " + h);
      I.startsWith("file://") && u("warning: Loading from a file URI (" + I + ") is not supported in most browsers. See https://emscripten.org/docs/getting_started/FAQ.html#how-do-i-run-a-local-webserver-for-testing-why-does-my-program-stall-in-downloading-or-preparing");
      n(h);
    });
  }
  var e = {env:Mf, wasi_snapshot_preview1:Mf};
  hb();
  var f = b;
  if (b.instantiateWasm) {
    try {
      return b.instantiateWasm(e, a);
    } catch (g) {
      return u("Module.instantiateWasm callback failed with error: " + g), !1;
    }
  }
  (function() {
    return ta || "function" !== typeof WebAssembly.instantiateStreaming || jb() || I.startsWith("file://") || "function" !== typeof fetch ? d(c) : fetch(I, {credentials:"same-origin"}).then(function(g) {
      return WebAssembly.instantiateStreaming(g, e).then(c, function(h) {
        u("wasm streaming compile failed: " + h);
        u("falling back to ArrayBuffer instantiation");
        return d(c);
      });
    });
  })().catch(ca);
  return {};
})();
b.___wasm_call_ctors = L("__wasm_call_ctors");
b._OrtInit = L("OrtInit");
b._OrtCreateSessionOptions = L("OrtCreateSessionOptions");
b._OrtAddSessionConfigEntry = L("OrtAddSessionConfigEntry");
b._OrtReleaseSessionOptions = L("OrtReleaseSessionOptions");
b._OrtCreateSession = L("OrtCreateSession");
b._OrtReleaseSession = L("OrtReleaseSession");
b._OrtGetInputCount = L("OrtGetInputCount");
b._OrtGetOutputCount = L("OrtGetOutputCount");
b._OrtGetInputName = L("OrtGetInputName");
b._OrtGetOutputName = L("OrtGetOutputName");
b._OrtFree = L("OrtFree");
b._OrtCreateTensor = L("OrtCreateTensor");
b._OrtGetTensorData = L("OrtGetTensorData");
b._OrtReleaseTensor = L("OrtReleaseTensor");
b._OrtCreateRunOptions = L("OrtCreateRunOptions");
b._OrtAddRunConfigEntry = L("OrtAddRunConfigEntry");
b._OrtReleaseRunOptions = L("OrtReleaseRunOptions");
b._OrtRun = L("OrtRun");
b._OrtEndProfiling = L("OrtEndProfiling");
var Ob = b.___errno_location = L("__errno_location"), Pa = b._malloc = L("malloc"), rb = b._free = L("free");
b.___funcs_on_exit = L("__funcs_on_exit");
b.___dl_seterr = L("__dl_seterr");
b._emscripten_main_thread_process_queued_calls = L("emscripten_main_thread_process_queued_calls");
var Fa = b._sbrk = L("sbrk"), Nb = b._memalign = L("memalign");
b._emscripten_get_sbrk_ptr = L("emscripten_get_sbrk_ptr");
var V = b._setThrew = L("setThrew"), Nf = b._emscripten_stack_init = function() {
  return (Nf = b._emscripten_stack_init = b.asm.emscripten_stack_init).apply(null, arguments);
};
b._emscripten_stack_get_free = function() {
  return (b._emscripten_stack_get_free = b.asm.emscripten_stack_get_free).apply(null, arguments);
};
var Ga = b._emscripten_stack_get_base = function() {
  return (Ga = b._emscripten_stack_get_base = b.asm.emscripten_stack_get_base).apply(null, arguments);
}, Wa = b._emscripten_stack_get_end = function() {
  return (Wa = b._emscripten_stack_get_end = b.asm.emscripten_stack_get_end).apply(null, arguments);
}, W = b.stackSave = L("stackSave"), Y = b.stackRestore = L("stackRestore"), Of = b.stackAlloc = L("stackAlloc");
b.___cxa_demangle = L("__cxa_demangle");
var Mb = b.___cxa_can_catch = L("__cxa_can_catch"), sb = b.___cxa_is_pointer_type = L("__cxa_is_pointer_type"), Pf = b.dynCall_ji = L("dynCall_ji"), Qf = b.dynCall_viijiiiiiiiii = L("dynCall_viijiiiiiiiii"), Rf = b.dynCall_viiij = L("dynCall_viiij"), Sf = b.dynCall_jii = L("dynCall_jii"), Tf = b.dynCall_jiji = L("dynCall_jiji"), Uf = b.dynCall_iiiiiij = L("dynCall_iiiiiij"), Vf = b.dynCall_iij = L("dynCall_iij"), Wf = b.dynCall_vij = L("dynCall_vij"), Xf = b.dynCall_viiijii = L("dynCall_viiijii"), 
Yf = b.dynCall_jj = L("dynCall_jj"), Zf = b.dynCall_viiijiii = L("dynCall_viiijiii"), $f = b.dynCall_ij = L("dynCall_ij"), ag = b.dynCall_viijj = L("dynCall_viijj"), bg = b.dynCall_iiiiijiii = L("dynCall_iiiiijiii"), cg = b.dynCall_viij = L("dynCall_viij"), dg = b.dynCall_iiiiijiiiii = L("dynCall_iiiiijiiiii"), eg = b.dynCall_iiiiiijji = L("dynCall_iiiiiijji"), fg = b.dynCall_vijiii = L("dynCall_vijiii"), gg = b.dynCall_jjj = L("dynCall_jjj"), hg = b.dynCall_viiji = L("dynCall_viiji"), ig = b.dynCall_viiiji = 
L("dynCall_viiiji"), jg = b.dynCall_viijjj = L("dynCall_viijjj"), kg = b.dynCall_viifiifijjjii = L("dynCall_viifiifijjjii"), lg = b.dynCall_jiii = L("dynCall_jiii"), mg = b.dynCall_vijiji = L("dynCall_vijiji"), ng = b.dynCall_jiij = L("dynCall_jiij"), og = b.dynCall_vjifiii = L("dynCall_vjifiii"), pg = b.dynCall_iiji = L("dynCall_iiji"), qg = b.dynCall_ijjj = L("dynCall_ijjj"), rg = b.dynCall_viijjiiiiiiiii = L("dynCall_viijjiiiiiiiii"), sg = b.dynCall_vijjjiiji = L("dynCall_vijjjiiji"), tg = b.dynCall_viijiiiiiiijjii = 
L("dynCall_viijiiiiiiijjii"), ug = b.dynCall_vjjfiii = L("dynCall_vjjfiii"), vg = b.dynCall_iiijj = L("dynCall_iiijj"), wg = b.dynCall_vijjjjiii = L("dynCall_vijjjjiii"), xg = b.dynCall_viiiijiiiiiiif = L("dynCall_viiiijiiiiiiif"), yg = b.dynCall_viijjjfiifiiii = L("dynCall_viijjjfiifiiii"), zg = b.dynCall_viiiiiiiijjj = L("dynCall_viiiiiiiijjj"), Ag = b.dynCall_viiiiiijjjjjii = L("dynCall_viiiiiijjjjjii"), Bg = b.dynCall_viiiijii = L("dynCall_viiiijii"), Cg = b.dynCall_viiiiij = L("dynCall_viiiiij"), 
Dg = b.dynCall_iji = L("dynCall_iji"), Eg = b.dynCall_vijjjjjjjjjjjjjii = L("dynCall_vijjjjjjjjjjjjjii"), Fg = b.dynCall_viiijjiiiiiii = L("dynCall_viiijjiiiiiii"), Gg = b.dynCall_viijiiiijii = L("dynCall_viijiiiijii"), Hg = b.dynCall_viifjjjijiiiii = L("dynCall_viifjjjijiiiii"), Ig = b.dynCall_viifjjijiiii = L("dynCall_viifjjijiiii"), Jg = b.dynCall_iiijiiiii = L("dynCall_iiijiiiii"), Kg = b.dynCall_vj = L("dynCall_vj"), Lg = b.dynCall_iiiiiji = L("dynCall_iiiiiji"), Mg = b.dynCall_vjiiiii = L("dynCall_vjiiiii"), 
Ng = b.dynCall_vjiiiiii = L("dynCall_vjiiiiii"), Og = b.dynCall_vijiiiiii = L("dynCall_vijiiiiii"), Pg = b.dynCall_vjiiiiiii = L("dynCall_vjiiiiiii"), Qg = b.dynCall_viijjjjjjjjjjjjjif = L("dynCall_viijjjjjjjjjjjjjif"), Rg = b.dynCall_viiiijj = L("dynCall_viiiijj"), Sg = b.dynCall_viiiiiji = L("dynCall_viiiiiji"), Tg = b.dynCall_j = L("dynCall_j"), Ug = b.dynCall_viiijjjii = L("dynCall_viiijjjii"), Vg = b.dynCall_iijj = L("dynCall_iijj"), Wg = b.dynCall_iiiij = L("dynCall_iiiij"), Xg = b.dynCall_viiijjjfffi = 
L("dynCall_viiijjjfffi"), Yg = b.dynCall_viiijiijjj = L("dynCall_viiijiijjj"), Zg = b.dynCall_viijjjj = L("dynCall_viijjjj"), $g = b.dynCall_vjjjjjjffiifiiiii = L("dynCall_vjjjjjjffiifiiiii"), ah = b.dynCall_vjjjjjjjjfffiifiiiii = L("dynCall_vjjjjjjjjfffiifiiiii"), bh = b.dynCall_jfi = L("dynCall_jfi"), ch = b.dynCall_fijjjjifi = L("dynCall_fijjjjifi"), dh = b.dynCall_vijjfffiii = L("dynCall_vijjfffiii"), eh = b.dynCall_vijiiiiiiii = L("dynCall_vijiiiiiiii"), fh = b.dynCall_viiijj = L("dynCall_viiijj"), 
gh = b.dynCall_viiiiijiiiiii = L("dynCall_viiiiijiiiiii"), hh = b.dynCall_viiiiijjiiiii = L("dynCall_viiiiijjiiiii"), ih = b.dynCall_viiiiji = L("dynCall_viiiiji"), jh = b.dynCall_viijjiii = L("dynCall_viijjiii"), kh = b.dynCall_vijii = L("dynCall_vijii"), lh = b.dynCall_iiiiji = L("dynCall_iiiiji"), mh = b.dynCall_viijjjjjjjjjjjjjii = L("dynCall_viijjjjjjjjjjjjjii"), nh = b.dynCall_viiiijiiiiiiii = L("dynCall_viiiijiiiiiiii"), oh = b.dynCall_iijjjf = L("dynCall_iijjjf"), ph = b.dynCall_viiiijjji = 
L("dynCall_viiiijjji");
b.dynCall_jjjjjj = L("dynCall_jjjjjj");
b.dynCall_jjjjjjj = L("dynCall_jjjjjjj");
var qh = b.dynCall_vijjjiij = L("dynCall_vijjjiij"), rh = b.dynCall_vijjj = L("dynCall_vijjj"), sh = b.dynCall_viiiiiij = L("dynCall_viiiiiij"), th = b.dynCall_viiiiiiij = L("dynCall_viiiiiiij"), uh = b.dynCall_viiijiiiiiiiii = L("dynCall_viiijiiiiiiiii"), vh = b.dynCall_iiiijjj = L("dynCall_iiiijjj"), wh = b.dynCall_viijiiii = L("dynCall_viijiiii"), xh = b.dynCall_iiijjii = L("dynCall_iiijjii");
b.dynCall_iijjii = L("dynCall_iijjii");
var yh = b.dynCall_vijjiiii = L("dynCall_vijjiiii"), zh = b.dynCall_viijjjiiiiiii = L("dynCall_viijjjiiiiiii"), Ah = b.dynCall_viijjjiiiii = L("dynCall_viijjjiiiii"), Bh = b.dynCall_viiijjjfii = L("dynCall_viiijjjfii"), Ch = b.dynCall_vijfjiiiii = L("dynCall_vijfjiiiii"), Dh = b.dynCall_iiiiiiiiijj = L("dynCall_iiiiiiiiijj"), Eh = b.dynCall_viiiiiiijiiiiiiiiiiiiiiiii = L("dynCall_viiiiiiijiiiiiiiiiiiiiiiii"), Fh = b.dynCall_ijiiii = L("dynCall_ijiiii"), Gh = b.dynCall_iiij = L("dynCall_iiij"), Hh = 
b.dynCall_iiiji = L("dynCall_iiiji"), Ih = b.dynCall_iiijii = L("dynCall_iiijii"), Jh = b.dynCall_iiiiiiiiiji = L("dynCall_iiiiiiiiiji"), Kh = b.dynCall_iiiiijji = L("dynCall_iiiiijji"), Lh = b.dynCall_iiiijjii = L("dynCall_iiiijjii"), Mh = b.dynCall_iiiijii = L("dynCall_iiiijii"), Nh = b.dynCall_iiijiii = L("dynCall_iiijiii"), Oh = b.dynCall_iiiiiiiiijii = L("dynCall_iiiiiiiiijii"), Ph = b.dynCall_iiiiiijjjii = L("dynCall_iiiiiijjjii"), Qh = b.dynCall_iiiiiiiijjji = L("dynCall_iiiiiiiijjji"), Rh = 
b.dynCall_iijiiii = L("dynCall_iijiiii"), Sh = b.dynCall_viiiij = L("dynCall_viiiij"), Th = b.dynCall_iijjjii = L("dynCall_iijjjii"), Uh = b.dynCall_jiiii = L("dynCall_jiiii"), Vh = b.dynCall_viijii = L("dynCall_viijii"), Wh = b.dynCall_viji = L("dynCall_viji"), Xh = b.dynCall_vjji = L("dynCall_vjji"), Yh = b.dynCall_ijii = L("dynCall_ijii"), Zh = b.dynCall_viiiiiiijiiii = L("dynCall_viiiiiiijiiii"), $h = b.dynCall_iiiiij = L("dynCall_iiiiij");
b.dynCall_iiiiijj = L("dynCall_iiiiijj");
b.dynCall_iiiiiijj = L("dynCall_iiiiiijj");
function ec(a, c, d) {
  var e = W();
  try {
    return N(a)(c, d);
  } catch (f) {
    Y(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    V(1, 0);
  }
}
function td(a, c) {
  var d = W();
  try {
    N(a)(c);
  } catch (e) {
    Y(d);
    if (e !== e + 0 && "longjmp" !== e) {
      throw e;
    }
    V(1, 0);
  }
}
function $b(a, c) {
  var d = W();
  try {
    return N(a)(c);
  } catch (e) {
    Y(d);
    if (e !== e + 0 && "longjmp" !== e) {
      throw e;
    }
    V(1, 0);
  }
}
function Dd(a, c, d) {
  var e = W();
  try {
    N(a)(c, d);
  } catch (f) {
    Y(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    V(1, 0);
  }
}
function Md(a, c, d, e) {
  var f = W();
  try {
    N(a)(c, d, e);
  } catch (g) {
    Y(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    V(1, 0);
  }
}
function hc(a, c, d, e) {
  var f = W();
  try {
    return N(a)(c, d, e);
  } catch (g) {
    Y(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    V(1, 0);
  }
}
function nc(a, c, d, e, f, g, h) {
  var k = W();
  try {
    return N(a)(c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function rd(a) {
  var c = W();
  try {
    N(a)();
  } catch (d) {
    Y(c);
    if (d !== d + 0 && "longjmp" !== d) {
      throw d;
    }
    V(1, 0);
  }
}
function Ud(a, c, d, e, f) {
  var g = W();
  try {
    N(a)(c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function mc(a, c, d, e, f, g) {
  var h = W();
  try {
    return N(a)(c, d, e, f, g);
  } catch (k) {
    Y(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    V(1, 0);
  }
}
function ic(a, c, d, e, f) {
  var g = W();
  try {
    return N(a)(c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function Xb(a) {
  var c = W();
  try {
    return N(a)();
  } catch (d) {
    Y(c);
    if (d !== d + 0 && "longjmp" !== d) {
      throw d;
    }
    V(1, 0);
  }
}
function Qb(a, c, d) {
  var e = W();
  try {
    return N(a)(c, d);
  } catch (f) {
    Y(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    V(1, 0);
  }
}
function he(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    N(a)(c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function oc(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    return N(a)(c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function Xd(a, c, d, e, f, g) {
  var h = W();
  try {
    N(a)(c, d, e, f, g);
  } catch (k) {
    Y(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    V(1, 0);
  }
}
function de(a, c, d, e, f, g, h) {
  var k = W();
  try {
    N(a)(c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function ke(a, c, d, e, f, g, h, k, l) {
  var m = W();
  try {
    N(a)(c, d, e, f, g, h, k, l);
  } catch (q) {
    Y(m);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    V(1, 0);
  }
}
function Tb(a, c) {
  var d = W();
  try {
    return N(a)(c);
  } catch (e) {
    Y(d);
    if (e !== e + 0 && "longjmp" !== e) {
      throw e;
    }
    V(1, 0);
  }
}
function wd(a, c, d) {
  var e = W();
  try {
    N(a)(c, d);
  } catch (f) {
    Y(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    V(1, 0);
  }
}
function Gd(a, c, d, e) {
  var f = W();
  try {
    N(a)(c, d, e);
  } catch (g) {
    Y(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    V(1, 0);
  }
}
function pc(a, c, d, e, f, g, h, k, l) {
  var m = W();
  try {
    return N(a)(c, d, e, f, g, h, k, l);
  } catch (q) {
    Y(m);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    V(1, 0);
  }
}
function me(a, c, d, e, f, g, h, k, l, m, q) {
  var r = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q);
  } catch (y) {
    Y(r);
    if (y !== y + 0 && "longjmp" !== y) {
      throw y;
    }
    V(1, 0);
  }
}
function qc(a, c, d, e, f, g, h, k, l, m) {
  var q = W();
  try {
    return N(a)(c, d, e, f, g, h, k, l, m);
  } catch (r) {
    Y(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    V(1, 0);
  }
}
function rc(a, c, d, e, f, g, h, k, l, m, q) {
  var r = W();
  try {
    return N(a)(c, d, e, f, g, h, k, l, m, q);
  } catch (y) {
    Y(r);
    if (y !== y + 0 && "longjmp" !== y) {
      throw y;
    }
    V(1, 0);
  }
}
function tc(a, c, d, e, f, g, h, k, l, m, q, r, y) {
  var p = W();
  try {
    return N(a)(c, d, e, f, g, h, k, l, m, q, r, y);
  } catch (w) {
    Y(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    V(1, 0);
  }
}
function ne(a, c, d, e, f, g, h, k, l, m, q, r) {
  var y = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r);
  } catch (p) {
    Y(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    V(1, 0);
  }
}
function pe(a, c, d, e, f, g, h, k, l, m, q, r, y, p) {
  var w = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r, y, p);
  } catch (z) {
    Y(w);
    if (z !== z + 0 && "longjmp" !== z) {
      throw z;
    }
    V(1, 0);
  }
}
function le(a, c, d, e, f, g, h, k, l, m) {
  var q = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m);
  } catch (r) {
    Y(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    V(1, 0);
  }
}
function qe(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w) {
  var z = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r, y, p, w);
  } catch (A) {
    Y(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    V(1, 0);
  }
}
function oe(a, c, d, e, f, g, h, k, l, m, q, r, y) {
  var p = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r, y);
  } catch (w) {
    Y(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    V(1, 0);
  }
}
function Ad(a, c, d, e, f, g, h, k, l, m) {
  var q = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m);
  } catch (r) {
    Y(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    V(1, 0);
  }
}
function Bd(a, c, d, e, f, g, h, k, l, m, q, r, y) {
  var p = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r, y);
  } catch (w) {
    Y(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    V(1, 0);
  }
}
function Fd(a, c, d, e, f) {
  var g = W();
  try {
    N(a)(c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function Zb(a, c, d, e) {
  var f = W();
  try {
    return N(a)(c, d, e);
  } catch (g) {
    Y(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    V(1, 0);
  }
}
function Vb(a, c, d, e) {
  var f = W();
  try {
    return N(a)(c, d, e);
  } catch (g) {
    Y(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    V(1, 0);
  }
}
function kc(a, c, d, e, f, g, h, k, l, m, q, r) {
  var y = W();
  try {
    return N(a)(c, d, e, f, g, h, k, l, m, q, r);
  } catch (p) {
    Y(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    V(1, 0);
  }
}
function se(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z) {
  var A = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r, y, p, w, z);
  } catch (C) {
    Y(A);
    if (C !== C + 0 && "longjmp" !== C) {
      throw C;
    }
    V(1, 0);
  }
}
function te(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A) {
  var C = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A);
  } catch (E) {
    Y(C);
    if (E !== E + 0 && "longjmp" !== E) {
      throw E;
    }
    V(1, 0);
  }
}
function Pd(a, c, d, e, f, g, h) {
  var k = W();
  try {
    N(a)(c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function ce(a, c, d, e, f, g, h, k, l, m, q, r, y) {
  var p = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r, y);
  } catch (w) {
    Y(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    V(1, 0);
  }
}
function Zd(a, c, d, e, f, g, h) {
  var k = W();
  try {
    N(a)(c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function Od(a, c, d, e, f, g) {
  var h = W();
  try {
    N(a)(c, d, e, f, g);
  } catch (k) {
    Y(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    V(1, 0);
  }
}
function ge(a, c, d, e, f, g, h, k, l) {
  var m = W();
  try {
    N(a)(c, d, e, f, g, h, k, l);
  } catch (q) {
    Y(m);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    V(1, 0);
  }
}
function $d(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    N(a)(c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function wc(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E) {
  var G = W();
  try {
    return N(a)(c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E);
  } catch (H) {
    Y(G);
    if (H !== H + 0 && "longjmp" !== H) {
      throw H;
    }
    V(1, 0);
  }
}
function cc(a, c, d) {
  var e = W();
  try {
    return N(a)(c, d);
  } catch (f) {
    Y(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    V(1, 0);
  }
}
function Yd(a, c, d, e, f, g, h, k, l, m, q, r) {
  var y = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r);
  } catch (p) {
    Y(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    V(1, 0);
  }
}
function Qd(a, c, d, e, f, g, h, k, l, m) {
  var q = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m);
  } catch (r) {
    Y(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    V(1, 0);
  }
}
function vc(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z) {
  var A = W();
  try {
    return N(a)(c, d, e, f, g, h, k, l, m, q, r, y, p, w, z);
  } catch (C) {
    Y(A);
    if (C !== C + 0 && "longjmp" !== C) {
      throw C;
    }
    V(1, 0);
  }
}
function gc(a, c, d, e, f, g) {
  var h = W();
  try {
    return N(a)(c, d, e, f, g);
  } catch (k) {
    Y(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    V(1, 0);
  }
}
function lc(a, c, d, e, f, g, h, k, l, m, q, r) {
  var y = W();
  try {
    return N(a)(c, d, e, f, g, h, k, l, m, q, r);
  } catch (p) {
    Y(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    V(1, 0);
  }
}
function re(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A) {
  var C = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A);
  } catch (E) {
    Y(C);
    if (E !== E + 0 && "longjmp" !== E) {
      throw E;
    }
    V(1, 0);
  }
}
function ae(a, c, d, e, f, g, h, k, l, m, q, r) {
  var y = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r);
  } catch (p) {
    Y(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    V(1, 0);
  }
}
function Wd(a, c, d, e, f, g, h, k, l, m, q, r, y) {
  var p = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r, y);
  } catch (w) {
    Y(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    V(1, 0);
  }
}
function Td(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z) {
  var A = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r, y, p, w, z);
  } catch (C) {
    Y(A);
    if (C !== C + 0 && "longjmp" !== C) {
      throw C;
    }
    V(1, 0);
  }
}
function sd(a, c, d, e, f) {
  var g = W();
  try {
    N(a)(c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function Hd(a, c, d, e, f) {
  var g = W();
  try {
    N(a)(c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function xc(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E, G) {
  var H = W();
  try {
    return N(a)(c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E, G);
  } catch (J) {
    Y(H);
    if (J !== J + 0 && "longjmp" !== J) {
      throw J;
    }
    V(1, 0);
  }
}
function be(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w) {
  var z = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r, y, p, w);
  } catch (A) {
    Y(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    V(1, 0);
  }
}
function Rb(a, c, d, e) {
  var f = W();
  try {
    return N(a)(c, d, e);
  } catch (g) {
    Y(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    V(1, 0);
  }
}
function yd(a, c, d, e) {
  var f = W();
  try {
    N(a)(c, d, e);
  } catch (g) {
    Y(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    V(1, 0);
  }
}
function fe(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    N(a)(c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function je(a, c, d, e, f, g, h, k, l, m, q, r, y) {
  var p = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r, y);
  } catch (w) {
    Y(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    V(1, 0);
  }
}
function Pb(a, c) {
  var d = W();
  try {
    return N(a)(c);
  } catch (e) {
    Y(d);
    if (e !== e + 0 && "longjmp" !== e) {
      throw e;
    }
    V(1, 0);
  }
}
function ee(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    N(a)(c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function ie(a, c, d, e, f, g, h, k, l, m, q, r, y) {
  var p = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r, y);
  } catch (w) {
    Y(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    V(1, 0);
  }
}
function Ub(a, c, d) {
  var e = W();
  try {
    return N(a)(c, d);
  } catch (f) {
    Y(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    V(1, 0);
  }
}
function sc(a, c, d, e, f, g, h, k, l, m, q, r) {
  var y = W();
  try {
    return N(a)(c, d, e, f, g, h, k, l, m, q, r);
  } catch (p) {
    Y(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    V(1, 0);
  }
}
function Cd(a, c, d, e, f, g) {
  var h = W();
  try {
    N(a)(c, d, e, f, g);
  } catch (k) {
    Y(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    V(1, 0);
  }
}
function zd(a, c, d, e, f) {
  var g = W();
  try {
    N(a)(c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function ud(a, c, d) {
  var e = W();
  try {
    N(a)(c, d);
  } catch (f) {
    Y(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    V(1, 0);
  }
}
function xd(a, c, d, e, f, g, h) {
  var k = W();
  try {
    N(a)(c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function ac(a, c, d, e) {
  var f = W();
  try {
    return N(a)(c, d, e);
  } catch (g) {
    Y(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    V(1, 0);
  }
}
function Ed(a, c, d, e) {
  var f = W();
  try {
    N(a)(c, d, e);
  } catch (g) {
    Y(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    V(1, 0);
  }
}
function uc(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w) {
  var z = W();
  try {
    return N(a)(c, d, e, f, g, h, k, l, m, q, r, y, p, w);
  } catch (A) {
    Y(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    V(1, 0);
  }
}
function Vd(a, c, d, e, f, g, h) {
  var k = W();
  try {
    N(a)(c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function Sb(a, c, d, e, f, g, h) {
  var k = W();
  try {
    return N(a)(c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function dc(a, c, d, e) {
  var f = W();
  try {
    return N(a)(c, d, e);
  } catch (g) {
    Y(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    V(1, 0);
  }
}
function Nd(a, c, d, e, f) {
  var g = W();
  try {
    N(a)(c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function ue(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E, G, H, J, O, P, M, R, S) {
  var T = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E, G, H, J, O, P, M, R, S);
  } catch (K) {
    Y(T);
    if (K !== K + 0 && "longjmp" !== K) {
      throw K;
    }
    V(1, 0);
  }
}
function bc(a, c, d, e) {
  var f = W();
  try {
    return N(a)(c, d, e);
  } catch (g) {
    Y(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    V(1, 0);
  }
}
function Yb(a, c) {
  var d = W();
  try {
    return N(a)(c);
  } catch (e) {
    Y(d);
    if (e !== e + 0 && "longjmp" !== e) {
      throw e;
    }
    V(1, 0);
  }
}
function fc(a, c, d, e, f) {
  var g = W();
  try {
    return N(a)(c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function vd(a, c, d, e) {
  var f = W();
  try {
    N(a)(c, d, e);
  } catch (g) {
    Y(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    V(1, 0);
  }
}
function Id(a, c, d, e, f) {
  var g = W();
  try {
    N(a)(c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function Sd(a, c, d, e, f, g, h, k, l, m) {
  var q = W();
  try {
    N(a)(c, d, e, f, g, h, k, l, m);
  } catch (r) {
    Y(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    V(1, 0);
  }
}
function Rd(a, c, d, e, f, g, h, k, l) {
  var m = W();
  try {
    N(a)(c, d, e, f, g, h, k, l);
  } catch (q) {
    Y(m);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    V(1, 0);
  }
}
function jc(a, c, d, e, f, g) {
  var h = W();
  try {
    return N(a)(c, d, e, f, g);
  } catch (k) {
    Y(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    V(1, 0);
  }
}
function od(a, c, d, e, f) {
  var g = W();
  try {
    return Tf(a, c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function Cc(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    return Uf(a, c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function kd(a, c, d) {
  var e = W();
  try {
    return Sf(a, c, d);
  } catch (f) {
    Y(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    V(1, 0);
  }
}
function jd(a, c) {
  var d = W();
  try {
    return Pf(a, c);
  } catch (e) {
    Y(d);
    if (e !== e + 0 && "longjmp" !== e) {
      throw e;
    }
    V(1, 0);
  }
}
function Wc(a, c, d, e) {
  var f = W();
  try {
    return Vf(a, c, d, e);
  } catch (g) {
    Y(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    V(1, 0);
  }
}
function of(a, c, d, e) {
  var f = W();
  try {
    Wf(a, c, d, e);
  } catch (g) {
    Y(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    V(1, 0);
  }
}
function qd(a, c, d, e, f) {
  var g = W();
  try {
    return gg(a, c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function Pe(a, c, d, e, f, g, h, k, l) {
  var m = W();
  try {
    Zf(a, c, d, e, f, g, h, k, l);
  } catch (q) {
    Y(m);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    V(1, 0);
  }
}
function Oe(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    Xf(a, c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function bd(a, c, d) {
  var e = W();
  try {
    return $f(a, c, d);
  } catch (f) {
    Y(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    V(1, 0);
  }
}
function df(a, c, d, e, f, g, h) {
  var k = W();
  try {
    ag(a, c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function Xe(a, c, d, e, f) {
  var g = W();
  try {
    cg(a, c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function Ic(a, c, d, e, f, g, h, k, l, m, q, r) {
  var y = W();
  try {
    return dg(a, c, d, e, f, g, h, k, l, m, q, r);
  } catch (p) {
    Y(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    V(1, 0);
  }
}
function Hc(a, c, d, e, f, g, h, k, l, m) {
  var q = W();
  try {
    return bg(a, c, d, e, f, g, h, k, l, m);
  } catch (r) {
    Y(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    V(1, 0);
  }
}
function Dc(a, c, d, e, f, g, h, k, l, m, q) {
  var r = W();
  try {
    return eg(a, c, d, e, f, g, h, k, l, m, q);
  } catch (y) {
    Y(r);
    if (y !== y + 0 && "longjmp" !== y) {
      throw y;
    }
    V(1, 0);
  }
}
function qf(a, c, d, e, f) {
  var g = W();
  try {
    Wh(a, c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function sf(a, c, d, e, f, g, h) {
  var k = W();
  try {
    fg(a, c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function Ye(a, c, d, e, f, g) {
  var h = W();
  try {
    hg(a, c, d, e, f, g);
  } catch (k) {
    Y(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    V(1, 0);
  }
}
function Ne(a, c, d, e, f, g, h) {
  var k = W();
  try {
    ig(a, c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function gf(a, c, d, e, f, g, h, k, l) {
  var m = W();
  try {
    jg(a, c, d, e, f, g, h, k, l);
  } catch (q) {
    Y(m);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    V(1, 0);
  }
}
function Jd(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z) {
  var A = W();
  try {
    kg(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z);
  } catch (C) {
    Y(A);
    if (C !== C + 0 && "longjmp" !== C) {
      throw C;
    }
    V(1, 0);
  }
}
function ld(a, c, d, e) {
  var f = W();
  try {
    return lg(a, c, d, e);
  } catch (g) {
    Y(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    V(1, 0);
  }
}
function vf(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    mg(a, c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function nd(a, c, d, e, f) {
  var g = W();
  try {
    return ng(a, c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function Ef(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    og(a, c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function Xc(a, c, d, e, f) {
  var g = W();
  try {
    return pg(a, c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function fd(a, c, d, e, f, g, h) {
  var k = W();
  try {
    return qg(a, c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function ff(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z) {
  var A = W();
  try {
    rg(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z);
  } catch (C) {
    Y(A);
    if (C !== C + 0 && "longjmp" !== C) {
      throw C;
    }
    V(1, 0);
  }
}
function Af(a, c, d, e, f, g, h, k, l, m, q, r, y) {
  var p = W();
  try {
    sg(a, c, d, e, f, g, h, k, l, m, q, r, y);
  } catch (w) {
    Y(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    V(1, 0);
  }
}
function bf(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C) {
  var E = W();
  try {
    tg(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C);
  } catch (G) {
    Y(E);
    if (G !== G + 0 && "longjmp" !== G) {
      throw G;
    }
    V(1, 0);
  }
}
function If(a, c, d, e, f, g, h, k, l) {
  var m = W();
  try {
    ug(a, c, d, e, f, g, h, k, l);
  } catch (q) {
    Y(m);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    V(1, 0);
  }
}
function Uc(a, c, d, e, f, g, h) {
  var k = W();
  try {
    return vg(a, c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function Bf(a, c, d, e, f, g, h, k, l, m, q, r, y) {
  var p = W();
  try {
    wg(a, c, d, e, f, g, h, k, l, m, q, r, y);
  } catch (w) {
    Y(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    V(1, 0);
  }
}
function Ie(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w) {
  var z = W();
  try {
    xg(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w);
  } catch (A) {
    Y(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    V(1, 0);
  }
}
function hf(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A) {
  var C = W();
  try {
    yg(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A);
  } catch (E) {
    Y(C);
    if (E !== E + 0 && "longjmp" !== E) {
      throw E;
    }
    V(1, 0);
  }
}
function ve(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w) {
  var z = W();
  try {
    zg(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w);
  } catch (A) {
    Y(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    V(1, 0);
  }
}
function Ae(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E) {
  var G = W();
  try {
    Ag(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E);
  } catch (H) {
    Y(G);
    if (H !== H + 0 && "longjmp" !== H) {
      throw H;
    }
    V(1, 0);
  }
}
function He(a, c, d, e, f, g, h, k, l) {
  var m = W();
  try {
    Bg(a, c, d, e, f, g, h, k, l);
  } catch (q) {
    Y(m);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    V(1, 0);
  }
}
function Be(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    Cg(a, c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function cd(a, c, d, e) {
  var f = W();
  try {
    return Dg(a, c, d, e);
  } catch (g) {
    Y(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    V(1, 0);
  }
}
function Cf(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E, G, H, J, O, P, M, R, S, T, K, U) {
  var sa = W();
  try {
    Eg(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E, G, H, J, O, P, M, R, S, T, K, U);
  } catch (Z) {
    Y(sa);
    if (Z !== Z + 0 && "longjmp" !== Z) {
      throw Z;
    }
    V(1, 0);
  }
}
function Te(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w) {
  var z = W();
  try {
    Fg(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w);
  } catch (A) {
    Y(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    V(1, 0);
  }
}
function cf(a, c, d, e, f, g, h, k, l, m, q, r, y) {
  var p = W();
  try {
    Gg(a, c, d, e, f, g, h, k, l, m, q, r, y);
  } catch (w) {
    Y(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    V(1, 0);
  }
}
function Ld(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C) {
  var E = W();
  try {
    Hg(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C);
  } catch (G) {
    Y(E);
    if (G !== G + 0 && "longjmp" !== G) {
      throw G;
    }
    V(1, 0);
  }
}
function Kd(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w) {
  var z = W();
  try {
    Ig(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w);
  } catch (A) {
    Y(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    V(1, 0);
  }
}
function Tc(a, c, d, e, f, g, h, k, l, m) {
  var q = W();
  try {
    return Jg(a, c, d, e, f, g, h, k, l, m);
  } catch (r) {
    Y(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    V(1, 0);
  }
}
function Df(a, c, d) {
  var e = W();
  try {
    Kg(a, c, d);
  } catch (f) {
    Y(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    V(1, 0);
  }
}
function Gc(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    return Lg(a, c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function Ff(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    Mg(a, c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function Gf(a, c, d, e, f, g, h, k, l) {
  var m = W();
  try {
    Ng(a, c, d, e, f, g, h, k, l);
  } catch (q) {
    Y(m);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    V(1, 0);
  }
}
function tf(a, c, d, e, f, g, h, k, l, m) {
  var q = W();
  try {
    Og(a, c, d, e, f, g, h, k, l, m);
  } catch (r) {
    Y(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    V(1, 0);
  }
}
function Hf(a, c, d, e, f, g, h, k, l, m) {
  var q = W();
  try {
    Pg(a, c, d, e, f, g, h, k, l, m);
  } catch (r) {
    Y(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    V(1, 0);
  }
}
function mf(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E, G, H, J, O, P, M, R, S, T, K, U, sa) {
  var Z = W();
  try {
    Qg(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E, G, H, J, O, P, M, R, S, T, K, U, sa);
  } catch (X) {
    Y(Z);
    if (X !== X + 0 && "longjmp" !== X) {
      throw X;
    }
    V(1, 0);
  }
}
function Me(a, c, d, e, f, g) {
  var h = W();
  try {
    Rf(a, c, d, e, f, g);
  } catch (k) {
    Y(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    V(1, 0);
  }
}
function Ke(a, c, d, e, f, g, h, k, l) {
  var m = W();
  try {
    Rg(a, c, d, e, f, g, h, k, l);
  } catch (q) {
    Y(m);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    V(1, 0);
  }
}
function Ce(a, c, d, e, f, g, h, k, l) {
  var m = W();
  try {
    Sg(a, c, d, e, f, g, h, k, l);
  } catch (q) {
    Y(m);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    V(1, 0);
  }
}
function We(a, c, d, e, f, g, h, k, l, m, q, r) {
  var y = W();
  try {
    Ug(a, c, d, e, f, g, h, k, l, m, q, r);
  } catch (p) {
    Y(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    V(1, 0);
  }
}
function Zc(a, c, d, e, f, g) {
  var h = W();
  try {
    return Vg(a, c, d, e, f, g);
  } catch (k) {
    Y(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    V(1, 0);
  }
}
function gd(a) {
  var c = W();
  try {
    return Tg(a);
  } catch (d) {
    Y(c);
    if (d !== d + 0 && "longjmp" !== d) {
      throw d;
    }
    V(1, 0);
  }
}
function Kc(a, c, d, e, f, g) {
  var h = W();
  try {
    return Wg(a, c, d, e, f, g);
  } catch (k) {
    Y(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    V(1, 0);
  }
}
function Ue(a, c, d, e, f, g, h, k, l, m, q, r, y, p) {
  var w = W();
  try {
    Xg(a, c, d, e, f, g, h, k, l, m, q, r, y, p);
  } catch (z) {
    Y(w);
    if (z !== z + 0 && "longjmp" !== z) {
      throw z;
    }
    V(1, 0);
  }
}
function Re(a, c, d, e, f, g, h, k, l, m, q, r, y, p) {
  var w = W();
  try {
    Yg(a, c, d, e, f, g, h, k, l, m, q, r, y, p);
  } catch (z) {
    Y(w);
    if (z !== z + 0 && "longjmp" !== z) {
      throw z;
    }
    V(1, 0);
  }
}
function lf(a, c, d, e, f, g, h, k, l, m, q) {
  var r = W();
  try {
    Zg(a, c, d, e, f, g, h, k, l, m, q);
  } catch (y) {
    Y(r);
    if (y !== y + 0 && "longjmp" !== y) {
      throw y;
    }
    V(1, 0);
  }
}
function Kf(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E, G, H, J, O) {
  var P = W();
  try {
    $g(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E, G, H, J, O);
  } catch (M) {
    Y(P);
    if (M !== M + 0 && "longjmp" !== M) {
      throw M;
    }
    V(1, 0);
  }
}
function Lf(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E, G, H, J, O, P, M, R, S, T) {
  var K = W();
  try {
    ah(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E, G, H, J, O, P, M, R, S, T);
  } catch (U) {
    Y(K);
    if (U !== U + 0 && "longjmp" !== U) {
      throw U;
    }
    V(1, 0);
  }
}
function Wb(a, c, d, e, f, g, h, k, l, m, q, r, y) {
  var p = W();
  try {
    return ch(a, c, d, e, f, g, h, k, l, m, q, r, y);
  } catch (w) {
    Y(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    V(1, 0);
  }
}
function wf(a, c, d, e, f, g, h, k, l, m, q, r) {
  var y = W();
  try {
    dh(a, c, d, e, f, g, h, k, l, m, q, r);
  } catch (p) {
    Y(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    V(1, 0);
  }
}
function uf(a, c, d, e, f, g, h, k, l, m, q, r) {
  var y = W();
  try {
    eh(a, c, d, e, f, g, h, k, l, m, q, r);
  } catch (p) {
    Y(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    V(1, 0);
  }
}
function hd(a, c, d) {
  var e = W();
  try {
    return bh(a, c, d);
  } catch (f) {
    Y(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    V(1, 0);
  }
}
function Se(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    fh(a, c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function De(a, c, d, e, f, g, h, k, l, m, q, r, y, p) {
  var w = W();
  try {
    gh(a, c, d, e, f, g, h, k, l, m, q, r, y, p);
  } catch (z) {
    Y(w);
    if (z !== z + 0 && "longjmp" !== z) {
      throw z;
    }
    V(1, 0);
  }
}
function Ee(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w) {
  var z = W();
  try {
    hh(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w);
  } catch (A) {
    Y(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    V(1, 0);
  }
}
function Ge(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    ih(a, c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function ef(a, c, d, e, f, g, h, k, l, m) {
  var q = W();
  try {
    jh(a, c, d, e, f, g, h, k, l, m);
  } catch (r) {
    Y(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    V(1, 0);
  }
}
function rf(a, c, d, e, f, g) {
  var h = W();
  try {
    kh(a, c, d, e, f, g);
  } catch (k) {
    Y(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    V(1, 0);
  }
}
function Lc(a, c, d, e, f, g, h) {
  var k = W();
  try {
    return lh(a, c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function nf(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E, G, H, J, O, P, M, R, S, T, K, U, sa) {
  var Z = W();
  try {
    mh(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E, G, H, J, O, P, M, R, S, T, K, U, sa);
  } catch (X) {
    Y(Z);
    if (X !== X + 0 && "longjmp" !== X) {
      throw X;
    }
    V(1, 0);
  }
}
function Je(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w) {
  var z = W();
  try {
    nh(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w);
  } catch (A) {
    Y(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    V(1, 0);
  }
}
function $c(a, c, d, e, f, g, h, k, l) {
  var m = W();
  try {
    return oh(a, c, d, e, f, g, h, k, l);
  } catch (q) {
    Y(m);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    V(1, 0);
  }
}
function Le(a, c, d, e, f, g, h, k, l, m, q, r) {
  var y = W();
  try {
    ph(a, c, d, e, f, g, h, k, l, m, q, r);
  } catch (p) {
    Y(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    V(1, 0);
  }
}
function zf(a, c, d, e, f, g, h, k, l, m, q, r) {
  var y = W();
  try {
    qh(a, c, d, e, f, g, h, k, l, m, q, r);
  } catch (p) {
    Y(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    V(1, 0);
  }
}
function yf(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    rh(a, c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function ze(a, c, d, e, f, g, h, k, l) {
  var m = W();
  try {
    sh(a, c, d, e, f, g, h, k, l);
  } catch (q) {
    Y(m);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    V(1, 0);
  }
}
function we(a, c, d, e, f, g, h, k, l, m) {
  var q = W();
  try {
    th(a, c, d, e, f, g, h, k, l, m);
  } catch (r) {
    Y(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    V(1, 0);
  }
}
function Qe(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w) {
  var z = W();
  try {
    uh(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w);
  } catch (A) {
    Y(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    V(1, 0);
  }
}
function Oc(a, c, d, e, f, g, h, k, l, m) {
  var q = W();
  try {
    return vh(a, c, d, e, f, g, h, k, l, m);
  } catch (r) {
    Y(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    V(1, 0);
  }
}
function $e(a, c, d, e, f, g, h, k, l) {
  var m = W();
  try {
    wh(a, c, d, e, f, g, h, k, l);
  } catch (q) {
    Y(m);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    V(1, 0);
  }
}
function Vc(a, c, d, e, f, g, h, k, l) {
  var m = W();
  try {
    return xh(a, c, d, e, f, g, h, k, l);
  } catch (q) {
    Y(m);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    V(1, 0);
  }
}
function xf(a, c, d, e, f, g, h, k, l, m) {
  var q = W();
  try {
    yh(a, c, d, e, f, g, h, k, l, m);
  } catch (r) {
    Y(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    V(1, 0);
  }
}
function kf(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z) {
  var A = W();
  try {
    zh(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z);
  } catch (C) {
    Y(A);
    if (C !== C + 0 && "longjmp" !== C) {
      throw C;
    }
    V(1, 0);
  }
}
function jf(a, c, d, e, f, g, h, k, l, m, q, r, y, p) {
  var w = W();
  try {
    Ah(a, c, d, e, f, g, h, k, l, m, q, r, y, p);
  } catch (z) {
    Y(w);
    if (z !== z + 0 && "longjmp" !== z) {
      throw z;
    }
    V(1, 0);
  }
}
function Ve(a, c, d, e, f, g, h, k, l, m, q, r, y) {
  var p = W();
  try {
    Bh(a, c, d, e, f, g, h, k, l, m, q, r, y);
  } catch (w) {
    Y(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    V(1, 0);
  }
}
function pf(a, c, d, e, f, g, h, k, l, m, q, r) {
  var y = W();
  try {
    Ch(a, c, d, e, f, g, h, k, l, m, q, r);
  } catch (p) {
    Y(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    V(1, 0);
  }
}
function Ac(a, c, d, e, f, g, h, k, l, m, q, r, y) {
  var p = W();
  try {
    return Dh(a, c, d, e, f, g, h, k, l, m, q, r, y);
  } catch (w) {
    Y(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    V(1, 0);
  }
}
function ye(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E, G, H, J, O, P, M, R, S) {
  var T = W();
  try {
    Eh(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w, z, A, C, E, G, H, J, O, P, M, R, S);
  } catch (K) {
    Y(T);
    if (K !== K + 0 && "longjmp" !== K) {
      throw K;
    }
    V(1, 0);
  }
}
function ed(a, c, d, e, f, g, h) {
  var k = W();
  try {
    return Fh(a, c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function af(a, c, d, e, f, g, h, k, l, m, q, r, y, p) {
  var w = W();
  try {
    Qf(a, c, d, e, f, g, h, k, l, m, q, r, y, p);
  } catch (z) {
    Y(w);
    if (z !== z + 0 && "longjmp" !== z) {
      throw z;
    }
    V(1, 0);
  }
}
function Pc(a, c, d, e, f) {
  var g = W();
  try {
    return Gh(a, c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function Qc(a, c, d, e, f, g) {
  var h = W();
  try {
    return Hh(a, c, d, e, f, g);
  } catch (k) {
    Y(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    V(1, 0);
  }
}
function Rc(a, c, d, e, f, g, h) {
  var k = W();
  try {
    return Ih(a, c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function yc(a, c, d, e, f, g, h, k, l, m, q, r) {
  var y = W();
  try {
    return Jh(a, c, d, e, f, g, h, k, l, m, q, r);
  } catch (p) {
    Y(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    V(1, 0);
  }
}
function Jc(a, c, d, e, f, g, h, k, l, m) {
  var q = W();
  try {
    return Kh(a, c, d, e, f, g, h, k, l, m);
  } catch (r) {
    Y(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    V(1, 0);
  }
}
function Nc(a, c, d, e, f, g, h, k, l, m) {
  var q = W();
  try {
    return Lh(a, c, d, e, f, g, h, k, l, m);
  } catch (r) {
    Y(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    V(1, 0);
  }
}
function Mc(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    return Mh(a, c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function Sc(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    return Nh(a, c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function zc(a, c, d, e, f, g, h, k, l, m, q, r, y) {
  var p = W();
  try {
    return Oh(a, c, d, e, f, g, h, k, l, m, q, r, y);
  } catch (w) {
    Y(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    V(1, 0);
  }
}
function Ec(a, c, d, e, f, g, h, k, l, m, q, r, y, p) {
  var w = W();
  try {
    return Ph(a, c, d, e, f, g, h, k, l, m, q, r, y, p);
  } catch (z) {
    Y(w);
    if (z !== z + 0 && "longjmp" !== z) {
      throw z;
    }
    V(1, 0);
  }
}
function Bc(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w) {
  var z = W();
  try {
    return Qh(a, c, d, e, f, g, h, k, l, m, q, r, y, p, w);
  } catch (A) {
    Y(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    V(1, 0);
  }
}
function Yc(a, c, d, e, f, g, h, k) {
  var l = W();
  try {
    return Rh(a, c, d, e, f, g, h, k);
  } catch (m) {
    Y(l);
    if (m !== m + 0 && "longjmp" !== m) {
      throw m;
    }
    V(1, 0);
  }
}
function Fe(a, c, d, e, f, g, h) {
  var k = W();
  try {
    Sh(a, c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function ad(a, c, d, e, f, g, h, k, l, m) {
  var q = W();
  try {
    return Th(a, c, d, e, f, g, h, k, l, m);
  } catch (r) {
    Y(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    V(1, 0);
  }
}
function pd(a, c, d) {
  var e = W();
  try {
    return Yf(a, c, d);
  } catch (f) {
    Y(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    V(1, 0);
  }
}
function md(a, c, d, e, f) {
  var g = W();
  try {
    return Uh(a, c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function Ze(a, c, d, e, f, g, h) {
  var k = W();
  try {
    Vh(a, c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
function Jf(a, c, d, e, f, g) {
  var h = W();
  try {
    Xh(a, c, d, e, f, g);
  } catch (k) {
    Y(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    V(1, 0);
  }
}
function dd(a, c, d, e, f) {
  var g = W();
  try {
    return Yh(a, c, d, e, f);
  } catch (h) {
    Y(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    V(1, 0);
  }
}
function xe(a, c, d, e, f, g, h, k, l, m, q, r, y, p) {
  var w = W();
  try {
    Zh(a, c, d, e, f, g, h, k, l, m, q, r, y, p);
  } catch (z) {
    Y(w);
    if (z !== z + 0 && "longjmp" !== z) {
      throw z;
    }
    V(1, 0);
  }
}
function Fc(a, c, d, e, f, g, h) {
  var k = W();
  try {
    return $h(a, c, d, e, f, g, h);
  } catch (l) {
    Y(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    V(1, 0);
  }
}
Object.getOwnPropertyDescriptor(b, "intArrayFromString") || (b.intArrayFromString = () => n("'intArrayFromString' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "intArrayToString") || (b.intArrayToString = () => n("'intArrayToString' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "ccall") || (b.ccall = () => n("'ccall' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "cwrap") || (b.cwrap = () => n("'cwrap' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "setValue") || (b.setValue = () => n("'setValue' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "getValue") || (b.getValue = () => n("'getValue' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "allocate") || (b.allocate = () => n("'allocate' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "UTF8ArrayToString") || (b.UTF8ArrayToString = () => n("'UTF8ArrayToString' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
b.UTF8ToString = F;
Object.getOwnPropertyDescriptor(b, "stringToUTF8Array") || (b.stringToUTF8Array = () => n("'stringToUTF8Array' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
b.stringToUTF8 = function(a, c, d) {
  assert("number" == typeof d, "stringToUTF8(str, outPtr, maxBytesToWrite) is missing the third parameter that specifies the length of the output buffer!");
  return Ma(a, La, c, d);
};
b.lengthBytesUTF8 = Na;
Object.getOwnPropertyDescriptor(b, "stackTrace") || (b.stackTrace = () => n("'stackTrace' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "addOnPreRun") || (b.addOnPreRun = () => n("'addOnPreRun' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "addOnInit") || (b.addOnInit = () => n("'addOnInit' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "addOnPreMain") || (b.addOnPreMain = () => n("'addOnPreMain' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "addOnExit") || (b.addOnExit = () => n("'addOnExit' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "addOnPostRun") || (b.addOnPostRun = () => n("'addOnPostRun' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "writeStringToMemory") || (b.writeStringToMemory = () => n("'writeStringToMemory' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "writeArrayToMemory") || (b.writeArrayToMemory = () => n("'writeArrayToMemory' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "writeAsciiToMemory") || (b.writeAsciiToMemory = () => n("'writeAsciiToMemory' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "addRunDependency") || (b.addRunDependency = () => n("'addRunDependency' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"));
Object.getOwnPropertyDescriptor(b, "removeRunDependency") || (b.removeRunDependency = () => n("'removeRunDependency' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"));
Object.getOwnPropertyDescriptor(b, "FS_createFolder") || (b.FS_createFolder = () => n("'FS_createFolder' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "FS_createPath") || (b.FS_createPath = () => n("'FS_createPath' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"));
Object.getOwnPropertyDescriptor(b, "FS_createDataFile") || (b.FS_createDataFile = () => n("'FS_createDataFile' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"));
Object.getOwnPropertyDescriptor(b, "FS_createPreloadedFile") || (b.FS_createPreloadedFile = () => n("'FS_createPreloadedFile' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"));
Object.getOwnPropertyDescriptor(b, "FS_createLazyFile") || (b.FS_createLazyFile = () => n("'FS_createLazyFile' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"));
Object.getOwnPropertyDescriptor(b, "FS_createLink") || (b.FS_createLink = () => n("'FS_createLink' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "FS_createDevice") || (b.FS_createDevice = () => n("'FS_createDevice' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"));
Object.getOwnPropertyDescriptor(b, "FS_unlink") || (b.FS_unlink = () => n("'FS_unlink' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"));
Object.getOwnPropertyDescriptor(b, "getLEB") || (b.getLEB = () => n("'getLEB' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "getFunctionTables") || (b.getFunctionTables = () => n("'getFunctionTables' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "alignFunctionTables") || (b.alignFunctionTables = () => n("'alignFunctionTables' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerFunctions") || (b.registerFunctions = () => n("'registerFunctions' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "addFunction") || (b.addFunction = () => n("'addFunction' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "removeFunction") || (b.removeFunction = () => n("'removeFunction' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "getFuncWrapper") || (b.getFuncWrapper = () => n("'getFuncWrapper' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "prettyPrint") || (b.prettyPrint = () => n("'prettyPrint' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "dynCall") || (b.dynCall = () => n("'dynCall' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "getCompilerSetting") || (b.getCompilerSetting = () => n("'getCompilerSetting' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "print") || (b.print = () => n("'print' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "printErr") || (b.printErr = () => n("'printErr' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "getTempRet0") || (b.getTempRet0 = () => n("'getTempRet0' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "setTempRet0") || (b.setTempRet0 = () => n("'setTempRet0' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "callMain") || (b.callMain = () => n("'callMain' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "abort") || (b.abort = () => n("'abort' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "keepRuntimeAlive") || (b.keepRuntimeAlive = () => n("'keepRuntimeAlive' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "zeroMemory") || (b.zeroMemory = () => n("'zeroMemory' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "stringToNewUTF8") || (b.stringToNewUTF8 = () => n("'stringToNewUTF8' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "setFileTime") || (b.setFileTime = () => n("'setFileTime' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "emscripten_realloc_buffer") || (b.emscripten_realloc_buffer = () => n("'emscripten_realloc_buffer' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "ENV") || (b.ENV = () => n("'ENV' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "withStackSave") || (b.withStackSave = () => n("'withStackSave' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "ERRNO_CODES") || (b.ERRNO_CODES = () => n("'ERRNO_CODES' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "ERRNO_MESSAGES") || (b.ERRNO_MESSAGES = () => n("'ERRNO_MESSAGES' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "setErrNo") || (b.setErrNo = () => n("'setErrNo' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "inetPton4") || (b.inetPton4 = () => n("'inetPton4' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "inetNtop4") || (b.inetNtop4 = () => n("'inetNtop4' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "inetPton6") || (b.inetPton6 = () => n("'inetPton6' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "inetNtop6") || (b.inetNtop6 = () => n("'inetNtop6' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "readSockaddr") || (b.readSockaddr = () => n("'readSockaddr' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "writeSockaddr") || (b.writeSockaddr = () => n("'writeSockaddr' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "DNS") || (b.DNS = () => n("'DNS' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "getHostByName") || (b.getHostByName = () => n("'getHostByName' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "GAI_ERRNO_MESSAGES") || (b.GAI_ERRNO_MESSAGES = () => n("'GAI_ERRNO_MESSAGES' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "Protocols") || (b.Protocols = () => n("'Protocols' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "Sockets") || (b.Sockets = () => n("'Sockets' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "getRandomDevice") || (b.getRandomDevice = () => n("'getRandomDevice' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "traverseStack") || (b.traverseStack = () => n("'traverseStack' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "convertFrameToPC") || (b.convertFrameToPC = () => n("'convertFrameToPC' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "UNWIND_CACHE") || (b.UNWIND_CACHE = () => n("'UNWIND_CACHE' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "saveInUnwindCache") || (b.saveInUnwindCache = () => n("'saveInUnwindCache' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "convertPCtoSourceLocation") || (b.convertPCtoSourceLocation = () => n("'convertPCtoSourceLocation' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "readAsmConstArgsArray") || (b.readAsmConstArgsArray = () => n("'readAsmConstArgsArray' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "readAsmConstArgs") || (b.readAsmConstArgs = () => n("'readAsmConstArgs' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "mainThreadEM_ASM") || (b.mainThreadEM_ASM = () => n("'mainThreadEM_ASM' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "jstoi_q") || (b.jstoi_q = () => n("'jstoi_q' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "jstoi_s") || (b.jstoi_s = () => n("'jstoi_s' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "getExecutableName") || (b.getExecutableName = () => n("'getExecutableName' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "listenOnce") || (b.listenOnce = () => n("'listenOnce' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "autoResumeAudioContext") || (b.autoResumeAudioContext = () => n("'autoResumeAudioContext' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "dynCallLegacy") || (b.dynCallLegacy = () => n("'dynCallLegacy' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "getDynCaller") || (b.getDynCaller = () => n("'getDynCaller' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "dynCall") || (b.dynCall = () => n("'dynCall' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "callRuntimeCallbacks") || (b.callRuntimeCallbacks = () => n("'callRuntimeCallbacks' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "wasmTableMirror") || (b.wasmTableMirror = () => n("'wasmTableMirror' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "setWasmTableEntry") || (b.setWasmTableEntry = () => n("'setWasmTableEntry' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "getWasmTableEntry") || (b.getWasmTableEntry = () => n("'getWasmTableEntry' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "handleException") || (b.handleException = () => n("'handleException' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "runtimeKeepalivePush") || (b.runtimeKeepalivePush = () => n("'runtimeKeepalivePush' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "runtimeKeepalivePop") || (b.runtimeKeepalivePop = () => n("'runtimeKeepalivePop' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "callUserCallback") || (b.callUserCallback = () => n("'callUserCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "maybeExit") || (b.maybeExit = () => n("'maybeExit' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "safeSetTimeout") || (b.safeSetTimeout = () => n("'safeSetTimeout' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "asmjsMangle") || (b.asmjsMangle = () => n("'asmjsMangle' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "asyncLoad") || (b.asyncLoad = () => n("'asyncLoad' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "alignMemory") || (b.alignMemory = () => n("'alignMemory' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "mmapAlloc") || (b.mmapAlloc = () => n("'mmapAlloc' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "reallyNegative") || (b.reallyNegative = () => n("'reallyNegative' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "unSign") || (b.unSign = () => n("'unSign' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "reSign") || (b.reSign = () => n("'reSign' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "formatString") || (b.formatString = () => n("'formatString' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "PATH") || (b.PATH = () => n("'PATH' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "PATH_FS") || (b.PATH_FS = () => n("'PATH_FS' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "SYSCALLS") || (b.SYSCALLS = () => n("'SYSCALLS' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "syscallMmap2") || (b.syscallMmap2 = () => n("'syscallMmap2' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "syscallMunmap") || (b.syscallMunmap = () => n("'syscallMunmap' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "getSocketFromFD") || (b.getSocketFromFD = () => n("'getSocketFromFD' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "getSocketAddress") || (b.getSocketAddress = () => n("'getSocketAddress' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "JSEvents") || (b.JSEvents = () => n("'JSEvents' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerKeyEventCallback") || (b.registerKeyEventCallback = () => n("'registerKeyEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "specialHTMLTargets") || (b.specialHTMLTargets = () => n("'specialHTMLTargets' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "maybeCStringToJsString") || (b.maybeCStringToJsString = () => n("'maybeCStringToJsString' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "findEventTarget") || (b.findEventTarget = () => n("'findEventTarget' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "findCanvasEventTarget") || (b.findCanvasEventTarget = () => n("'findCanvasEventTarget' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "getBoundingClientRect") || (b.getBoundingClientRect = () => n("'getBoundingClientRect' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "fillMouseEventData") || (b.fillMouseEventData = () => n("'fillMouseEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerMouseEventCallback") || (b.registerMouseEventCallback = () => n("'registerMouseEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerWheelEventCallback") || (b.registerWheelEventCallback = () => n("'registerWheelEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerUiEventCallback") || (b.registerUiEventCallback = () => n("'registerUiEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerFocusEventCallback") || (b.registerFocusEventCallback = () => n("'registerFocusEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "fillDeviceOrientationEventData") || (b.fillDeviceOrientationEventData = () => n("'fillDeviceOrientationEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerDeviceOrientationEventCallback") || (b.registerDeviceOrientationEventCallback = () => n("'registerDeviceOrientationEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "fillDeviceMotionEventData") || (b.fillDeviceMotionEventData = () => n("'fillDeviceMotionEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerDeviceMotionEventCallback") || (b.registerDeviceMotionEventCallback = () => n("'registerDeviceMotionEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "screenOrientation") || (b.screenOrientation = () => n("'screenOrientation' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "fillOrientationChangeEventData") || (b.fillOrientationChangeEventData = () => n("'fillOrientationChangeEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerOrientationChangeEventCallback") || (b.registerOrientationChangeEventCallback = () => n("'registerOrientationChangeEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "fillFullscreenChangeEventData") || (b.fillFullscreenChangeEventData = () => n("'fillFullscreenChangeEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerFullscreenChangeEventCallback") || (b.registerFullscreenChangeEventCallback = () => n("'registerFullscreenChangeEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerRestoreOldStyle") || (b.registerRestoreOldStyle = () => n("'registerRestoreOldStyle' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "hideEverythingExceptGivenElement") || (b.hideEverythingExceptGivenElement = () => n("'hideEverythingExceptGivenElement' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "restoreHiddenElements") || (b.restoreHiddenElements = () => n("'restoreHiddenElements' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "setLetterbox") || (b.setLetterbox = () => n("'setLetterbox' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "currentFullscreenStrategy") || (b.currentFullscreenStrategy = () => n("'currentFullscreenStrategy' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "restoreOldWindowedStyle") || (b.restoreOldWindowedStyle = () => n("'restoreOldWindowedStyle' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "softFullscreenResizeWebGLRenderTarget") || (b.softFullscreenResizeWebGLRenderTarget = () => n("'softFullscreenResizeWebGLRenderTarget' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "doRequestFullscreen") || (b.doRequestFullscreen = () => n("'doRequestFullscreen' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "fillPointerlockChangeEventData") || (b.fillPointerlockChangeEventData = () => n("'fillPointerlockChangeEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerPointerlockChangeEventCallback") || (b.registerPointerlockChangeEventCallback = () => n("'registerPointerlockChangeEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerPointerlockErrorEventCallback") || (b.registerPointerlockErrorEventCallback = () => n("'registerPointerlockErrorEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "requestPointerLock") || (b.requestPointerLock = () => n("'requestPointerLock' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "fillVisibilityChangeEventData") || (b.fillVisibilityChangeEventData = () => n("'fillVisibilityChangeEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerVisibilityChangeEventCallback") || (b.registerVisibilityChangeEventCallback = () => n("'registerVisibilityChangeEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerTouchEventCallback") || (b.registerTouchEventCallback = () => n("'registerTouchEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "fillGamepadEventData") || (b.fillGamepadEventData = () => n("'fillGamepadEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerGamepadEventCallback") || (b.registerGamepadEventCallback = () => n("'registerGamepadEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerBeforeUnloadEventCallback") || (b.registerBeforeUnloadEventCallback = () => n("'registerBeforeUnloadEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "fillBatteryEventData") || (b.fillBatteryEventData = () => n("'fillBatteryEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "battery") || (b.battery = () => n("'battery' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "registerBatteryEventCallback") || (b.registerBatteryEventCallback = () => n("'registerBatteryEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "setCanvasElementSize") || (b.setCanvasElementSize = () => n("'setCanvasElementSize' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "getCanvasElementSize") || (b.getCanvasElementSize = () => n("'getCanvasElementSize' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "demangle") || (b.demangle = () => n("'demangle' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "demangleAll") || (b.demangleAll = () => n("'demangleAll' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "jsStackTrace") || (b.jsStackTrace = () => n("'jsStackTrace' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "stackTrace") || (b.stackTrace = () => n("'stackTrace' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "getEnvStrings") || (b.getEnvStrings = () => n("'getEnvStrings' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "checkWasiClock") || (b.checkWasiClock = () => n("'checkWasiClock' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "flush_NO_FILESYSTEM") || (b.flush_NO_FILESYSTEM = () => n("'flush_NO_FILESYSTEM' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "writeI53ToI64") || (b.writeI53ToI64 = () => n("'writeI53ToI64' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "writeI53ToI64Clamped") || (b.writeI53ToI64Clamped = () => n("'writeI53ToI64Clamped' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "writeI53ToI64Signaling") || (b.writeI53ToI64Signaling = () => n("'writeI53ToI64Signaling' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "writeI53ToU64Clamped") || (b.writeI53ToU64Clamped = () => n("'writeI53ToU64Clamped' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "writeI53ToU64Signaling") || (b.writeI53ToU64Signaling = () => n("'writeI53ToU64Signaling' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "readI53FromI64") || (b.readI53FromI64 = () => n("'readI53FromI64' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "readI53FromU64") || (b.readI53FromU64 = () => n("'readI53FromU64' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "convertI32PairToI53") || (b.convertI32PairToI53 = () => n("'convertI32PairToI53' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "convertU32PairToI53") || (b.convertU32PairToI53 = () => n("'convertU32PairToI53' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "setImmediateWrapped") || (b.setImmediateWrapped = () => n("'setImmediateWrapped' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "clearImmediateWrapped") || (b.clearImmediateWrapped = () => n("'clearImmediateWrapped' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "polyfillSetImmediate") || (b.polyfillSetImmediate = () => n("'polyfillSetImmediate' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "uncaughtExceptionCount") || (b.uncaughtExceptionCount = () => n("'uncaughtExceptionCount' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "exceptionLast") || (b.exceptionLast = () => n("'exceptionLast' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "exceptionCaught") || (b.exceptionCaught = () => n("'exceptionCaught' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "ExceptionInfo") || (b.ExceptionInfo = () => n("'ExceptionInfo' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "CatchInfo") || (b.CatchInfo = () => n("'CatchInfo' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "exception_addRef") || (b.exception_addRef = () => n("'exception_addRef' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "exception_decRef") || (b.exception_decRef = () => n("'exception_decRef' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "Browser") || (b.Browser = () => n("'Browser' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "funcWrappers") || (b.funcWrappers = () => n("'funcWrappers' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "getFuncWrapper") || (b.getFuncWrapper = () => n("'getFuncWrapper' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "setMainLoop") || (b.setMainLoop = () => n("'setMainLoop' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "wget") || (b.wget = () => n("'wget' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "tempFixedLengthArray") || (b.tempFixedLengthArray = () => n("'tempFixedLengthArray' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "miniTempWebGLFloatBuffers") || (b.miniTempWebGLFloatBuffers = () => n("'miniTempWebGLFloatBuffers' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "heapObjectForWebGLType") || (b.heapObjectForWebGLType = () => n("'heapObjectForWebGLType' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "heapAccessShiftForWebGLHeap") || (b.heapAccessShiftForWebGLHeap = () => n("'heapAccessShiftForWebGLHeap' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "GL") || (b.GL = () => n("'GL' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "emscriptenWebGLGet") || (b.emscriptenWebGLGet = () => n("'emscriptenWebGLGet' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "computeUnpackAlignedImageSize") || (b.computeUnpackAlignedImageSize = () => n("'computeUnpackAlignedImageSize' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "emscriptenWebGLGetTexPixelData") || (b.emscriptenWebGLGetTexPixelData = () => n("'emscriptenWebGLGetTexPixelData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "emscriptenWebGLGetUniform") || (b.emscriptenWebGLGetUniform = () => n("'emscriptenWebGLGetUniform' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "webglGetUniformLocation") || (b.webglGetUniformLocation = () => n("'webglGetUniformLocation' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "webglPrepareUniformLocationsBeforeFirstUse") || (b.webglPrepareUniformLocationsBeforeFirstUse = () => n("'webglPrepareUniformLocationsBeforeFirstUse' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "webglGetLeftBracePos") || (b.webglGetLeftBracePos = () => n("'webglGetLeftBracePos' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "emscriptenWebGLGetVertexAttrib") || (b.emscriptenWebGLGetVertexAttrib = () => n("'emscriptenWebGLGetVertexAttrib' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "writeGLArray") || (b.writeGLArray = () => n("'writeGLArray' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "AL") || (b.AL = () => n("'AL' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "SDL_unicode") || (b.SDL_unicode = () => n("'SDL_unicode' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "SDL_ttfContext") || (b.SDL_ttfContext = () => n("'SDL_ttfContext' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "SDL_audio") || (b.SDL_audio = () => n("'SDL_audio' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "SDL") || (b.SDL = () => n("'SDL' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "SDL_gfx") || (b.SDL_gfx = () => n("'SDL_gfx' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "GLUT") || (b.GLUT = () => n("'GLUT' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "EGL") || (b.EGL = () => n("'EGL' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "GLFW_Window") || (b.GLFW_Window = () => n("'GLFW_Window' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "GLFW") || (b.GLFW = () => n("'GLFW' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "GLEW") || (b.GLEW = () => n("'GLEW' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "IDBStore") || (b.IDBStore = () => n("'IDBStore' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "runAndAbortIfError") || (b.runAndAbortIfError = () => n("'runAndAbortIfError' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "warnOnce") || (b.warnOnce = () => n("'warnOnce' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
b.stackSave = W;
b.stackRestore = Y;
b.stackAlloc = Of;
Object.getOwnPropertyDescriptor(b, "AsciiToString") || (b.AsciiToString = () => n("'AsciiToString' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "stringToAscii") || (b.stringToAscii = () => n("'stringToAscii' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "UTF16ToString") || (b.UTF16ToString = () => n("'UTF16ToString' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "stringToUTF16") || (b.stringToUTF16 = () => n("'stringToUTF16' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "lengthBytesUTF16") || (b.lengthBytesUTF16 = () => n("'lengthBytesUTF16' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "UTF32ToString") || (b.UTF32ToString = () => n("'UTF32ToString' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "stringToUTF32") || (b.stringToUTF32 = () => n("'stringToUTF32' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "lengthBytesUTF32") || (b.lengthBytesUTF32 = () => n("'lengthBytesUTF32' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "allocateUTF8") || (b.allocateUTF8 = () => n("'allocateUTF8' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(b, "allocateUTF8OnStack") || (b.allocateUTF8OnStack = () => n("'allocateUTF8OnStack' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
b.writeStackCookie = Va;
b.checkStackCookie = Xa;
Object.getOwnPropertyDescriptor(b, "ALLOC_NORMAL") || Object.defineProperty(b, "ALLOC_NORMAL", {configurable:!0, get:function() {
  n("'ALLOC_NORMAL' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)");
}});
Object.getOwnPropertyDescriptor(b, "ALLOC_STACK") || Object.defineProperty(b, "ALLOC_STACK", {configurable:!0, get:function() {
  n("'ALLOC_STACK' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)");
}});
var ai;
fb = function bi() {
  ai || ci();
  ai || (fb = bi);
};
function ci() {
  function a() {
    if (!ai && (ai = !0, b.calledRun = !0, !Ia)) {
      Xa();
      assert(!Da);
      Da = !0;
      nb(ab);
      ba(b);
      if (b.onRuntimeInitialized) {
        b.onRuntimeInitialized();
      }
      assert(!b._main, 'compiled without a main, but one is present. if you added it from JS, use Module["onRuntimeInitialized"]');
      Xa();
      if (b.postRun) {
        for ("function" == typeof b.postRun && (b.postRun = [b.postRun]); b.postRun.length;) {
          var c = b.postRun.shift();
          bb.unshift(c);
        }
      }
      nb(bb);
    }
  }
  if (!(0 < db)) {
    Nf();
    Va();
    if (b.preRun) {
      for ("function" == typeof b.preRun && (b.preRun = [b.preRun]); b.preRun.length;) {
        cb();
      }
    }
    nb($a);
    0 < db || (b.setStatus ? (b.setStatus("Running..."), setTimeout(function() {
      setTimeout(function() {
        b.setStatus("");
      }, 1);
      a();
    }, 1)) : a(), Xa());
  }
}
b.run = ci;
if (b.preInit) {
  for ("function" == typeof b.preInit && (b.preInit = [b.preInit]); 0 < b.preInit.length;) {
    b.preInit.pop()();
  }
}
ci();



  return ortWasm.ready
}
);
})();
if (typeof exports === 'object' && typeof module === 'object')
  module.exports = ortWasm;
else if (typeof define === 'function' && define['amd'])
  define([], function() { return ortWasm; });
else if (typeof exports === 'object')
  exports["ortWasm"] = ortWasm;
