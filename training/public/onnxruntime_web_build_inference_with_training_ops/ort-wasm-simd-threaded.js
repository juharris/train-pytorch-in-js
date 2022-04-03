
var ortWasmSimdThreaded = (() => {
  var _scriptDir = typeof document !== 'undefined' && document.currentScript ? document.currentScript.src : undefined;
  if (typeof __filename !== 'undefined') _scriptDir = _scriptDir || __filename;
  return (
function(ortWasmSimdThreaded) {
  ortWasmSimdThreaded = ortWasmSimdThreaded || {};


function ba() {
  c.buffer != m && da(c.buffer);
  return ea;
}
function ha() {
  c.buffer != m && da(c.buffer);
  return ia;
}
function ka() {
  c.buffer != m && da(c.buffer);
  return la;
}
function t() {
  c.buffer != m && da(c.buffer);
  return ma;
}
function na() {
  c.buffer != m && da(c.buffer);
  return oa;
}
function pa() {
  c.buffer != m && da(c.buffer);
  return qa;
}
var u;
u || (u = typeof ortWasmSimdThreaded !== 'undefined' ? ortWasmSimdThreaded : {});
var ra = Object.assign, sa, ta;
u.ready = new Promise(function(a, b) {
  sa = a;
  ta = b;
});
Object.getOwnPropertyDescriptor(u.ready, "__emscripten_thread_init") || (Object.defineProperty(u.ready, "__emscripten_thread_init", {configurable:!0, get:function() {
  v("You are getting __emscripten_thread_init on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "__emscripten_thread_init", {configurable:!0, set:function() {
  v("You are setting __emscripten_thread_init on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "__emscripten_thread_exit") || (Object.defineProperty(u.ready, "__emscripten_thread_exit", {configurable:!0, get:function() {
  v("You are getting __emscripten_thread_exit on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "__emscripten_thread_exit", {configurable:!0, set:function() {
  v("You are setting __emscripten_thread_exit on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_emscripten_tls_init") || (Object.defineProperty(u.ready, "_emscripten_tls_init", {configurable:!0, get:function() {
  v("You are getting _emscripten_tls_init on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_emscripten_tls_init", {configurable:!0, set:function() {
  v("You are setting _emscripten_tls_init on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_emscripten_current_thread_process_queued_calls") || (Object.defineProperty(u.ready, "_emscripten_current_thread_process_queued_calls", {configurable:!0, get:function() {
  v("You are getting _emscripten_current_thread_process_queued_calls on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_emscripten_current_thread_process_queued_calls", {configurable:!0, set:function() {
  v("You are setting _emscripten_current_thread_process_queued_calls on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_pthread_self") || (Object.defineProperty(u.ready, "_pthread_self", {configurable:!0, get:function() {
  v("You are getting _pthread_self on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_pthread_self", {configurable:!0, set:function() {
  v("You are setting _pthread_self on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "establishStackSpace") || (Object.defineProperty(u.ready, "establishStackSpace", {configurable:!0, get:function() {
  v("You are getting establishStackSpace on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "establishStackSpace", {configurable:!0, set:function() {
  v("You are setting establishStackSpace on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "invokeEntryPoint") || (Object.defineProperty(u.ready, "invokeEntryPoint", {configurable:!0, get:function() {
  v("You are getting invokeEntryPoint on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "invokeEntryPoint", {configurable:!0, set:function() {
  v("You are setting invokeEntryPoint on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtInit") || (Object.defineProperty(u.ready, "_OrtInit", {configurable:!0, get:function() {
  v("You are getting _OrtInit on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtInit", {configurable:!0, set:function() {
  v("You are setting _OrtInit on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtCreateSessionOptions") || (Object.defineProperty(u.ready, "_OrtCreateSessionOptions", {configurable:!0, get:function() {
  v("You are getting _OrtCreateSessionOptions on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtCreateSessionOptions", {configurable:!0, set:function() {
  v("You are setting _OrtCreateSessionOptions on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtAddSessionConfigEntry") || (Object.defineProperty(u.ready, "_OrtAddSessionConfigEntry", {configurable:!0, get:function() {
  v("You are getting _OrtAddSessionConfigEntry on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtAddSessionConfigEntry", {configurable:!0, set:function() {
  v("You are setting _OrtAddSessionConfigEntry on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtReleaseSessionOptions") || (Object.defineProperty(u.ready, "_OrtReleaseSessionOptions", {configurable:!0, get:function() {
  v("You are getting _OrtReleaseSessionOptions on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtReleaseSessionOptions", {configurable:!0, set:function() {
  v("You are setting _OrtReleaseSessionOptions on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtCreateSession") || (Object.defineProperty(u.ready, "_OrtCreateSession", {configurable:!0, get:function() {
  v("You are getting _OrtCreateSession on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtCreateSession", {configurable:!0, set:function() {
  v("You are setting _OrtCreateSession on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtReleaseSession") || (Object.defineProperty(u.ready, "_OrtReleaseSession", {configurable:!0, get:function() {
  v("You are getting _OrtReleaseSession on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtReleaseSession", {configurable:!0, set:function() {
  v("You are setting _OrtReleaseSession on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtGetInputCount") || (Object.defineProperty(u.ready, "_OrtGetInputCount", {configurable:!0, get:function() {
  v("You are getting _OrtGetInputCount on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtGetInputCount", {configurable:!0, set:function() {
  v("You are setting _OrtGetInputCount on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtGetOutputCount") || (Object.defineProperty(u.ready, "_OrtGetOutputCount", {configurable:!0, get:function() {
  v("You are getting _OrtGetOutputCount on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtGetOutputCount", {configurable:!0, set:function() {
  v("You are setting _OrtGetOutputCount on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtGetInputName") || (Object.defineProperty(u.ready, "_OrtGetInputName", {configurable:!0, get:function() {
  v("You are getting _OrtGetInputName on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtGetInputName", {configurable:!0, set:function() {
  v("You are setting _OrtGetInputName on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtGetOutputName") || (Object.defineProperty(u.ready, "_OrtGetOutputName", {configurable:!0, get:function() {
  v("You are getting _OrtGetOutputName on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtGetOutputName", {configurable:!0, set:function() {
  v("You are setting _OrtGetOutputName on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtFree") || (Object.defineProperty(u.ready, "_OrtFree", {configurable:!0, get:function() {
  v("You are getting _OrtFree on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtFree", {configurable:!0, set:function() {
  v("You are setting _OrtFree on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtCreateTensor") || (Object.defineProperty(u.ready, "_OrtCreateTensor", {configurable:!0, get:function() {
  v("You are getting _OrtCreateTensor on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtCreateTensor", {configurable:!0, set:function() {
  v("You are setting _OrtCreateTensor on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtGetTensorData") || (Object.defineProperty(u.ready, "_OrtGetTensorData", {configurable:!0, get:function() {
  v("You are getting _OrtGetTensorData on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtGetTensorData", {configurable:!0, set:function() {
  v("You are setting _OrtGetTensorData on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtReleaseTensor") || (Object.defineProperty(u.ready, "_OrtReleaseTensor", {configurable:!0, get:function() {
  v("You are getting _OrtReleaseTensor on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtReleaseTensor", {configurable:!0, set:function() {
  v("You are setting _OrtReleaseTensor on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtCreateRunOptions") || (Object.defineProperty(u.ready, "_OrtCreateRunOptions", {configurable:!0, get:function() {
  v("You are getting _OrtCreateRunOptions on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtCreateRunOptions", {configurable:!0, set:function() {
  v("You are setting _OrtCreateRunOptions on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtAddRunConfigEntry") || (Object.defineProperty(u.ready, "_OrtAddRunConfigEntry", {configurable:!0, get:function() {
  v("You are getting _OrtAddRunConfigEntry on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtAddRunConfigEntry", {configurable:!0, set:function() {
  v("You are setting _OrtAddRunConfigEntry on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtReleaseRunOptions") || (Object.defineProperty(u.ready, "_OrtReleaseRunOptions", {configurable:!0, get:function() {
  v("You are getting _OrtReleaseRunOptions on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtReleaseRunOptions", {configurable:!0, set:function() {
  v("You are setting _OrtReleaseRunOptions on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtRun") || (Object.defineProperty(u.ready, "_OrtRun", {configurable:!0, get:function() {
  v("You are getting _OrtRun on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtRun", {configurable:!0, set:function() {
  v("You are setting _OrtRun on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "_OrtEndProfiling") || (Object.defineProperty(u.ready, "_OrtEndProfiling", {configurable:!0, get:function() {
  v("You are getting _OrtEndProfiling on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "_OrtEndProfiling", {configurable:!0, set:function() {
  v("You are setting _OrtEndProfiling on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
Object.getOwnPropertyDescriptor(u.ready, "onRuntimeInitialized") || (Object.defineProperty(u.ready, "onRuntimeInitialized", {configurable:!0, get:function() {
  v("You are getting onRuntimeInitialized on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}), Object.defineProperty(u.ready, "onRuntimeInitialized", {configurable:!0, set:function() {
  v("You are setting onRuntimeInitialized on the Promise object, instead of the instance. Use .then() to get called back with the instance, see the MODULARIZE docs in src/settings.js");
}}));
var ua = ra({}, u), va = "./this.program", wa = (a, b) => {
  throw b;
}, xa = "object" === typeof window, ya = "function" === typeof importScripts, x = "object" === typeof process && "object" === typeof process.versions && "string" === typeof process.versions.node, za = !xa && !x && !ya;
if (u.ENVIRONMENT) {
  throw Error("Module.ENVIRONMENT has been deprecated. To force the environment, use the ENVIRONMENT compile-time option (for example, -s ENVIRONMENT=web or -s ENVIRONMENT=node)");
}
var B = u.ENVIRONMENT_IS_PTHREAD || !1, D = "";
function Aa(a) {
  return u.locateFile ? u.locateFile(a, D) : D + a;
}
var Ba, Ca, Da;
function Ea(a) {
  if (!(a instanceof Fa)) {
    var b = a;
    a && "object" === typeof a && a.stack && (b = [a, a.stack]);
    E("exiting due to exception: " + b);
  }
}
var fs, Ha, Ia;
if (x) {
  if ("object" !== typeof process || "function" !== typeof require) {
    throw Error("not compiled for this environment (did you build to HTML and try to run it not on the web, or set ENVIRONMENT to something - like node - and run it someplace else - like on the web?)");
  }
  D = ya ? require("path").dirname(D) + "/" : __dirname + "/";
  Ia = () => {
    Ha || (fs = require("fs"), Ha = require("path"));
  };
  Ba = function(b, d) {
    Ia();
    b = Ha.normalize(b);
    return fs.readFileSync(b, d ? null : "utf8");
  };
  Da = b => {
    b = Ba(b, !0);
    b.buffer || (b = new Uint8Array(b));
    assert(b.buffer);
    return b;
  };
  Ca = (b, d, e) => {
    Ia();
    b = Ha.normalize(b);
    fs.readFile(b, function(f, g) {
      f ? e(f) : d(g.buffer);
    });
  };
  1 < process.argv.length && (va = process.argv[1].replace(/\\/g, "/"));
  process.argv.slice(2);
  process.on("uncaughtException", function(b) {
    if (!(b instanceof Fa)) {
      throw b;
    }
  });
  process.on("unhandledRejection", function(b) {
    throw b;
  });
  wa = (b, d) => {
    if (Ja()) {
      throw process.exitCode = b, d;
    }
    Ea(d);
    process.exit(b);
  };
  u.inspect = function() {
    return "[Emscripten Module object]";
  };
  let a;
  try {
    a = require("worker_threads");
  } catch (b) {
    throw console.error('The "worker_threads" module is not supported in this node.js build - perhaps a newer version is needed?'), b;
  }
  global.Worker = a.Worker;
} else if (za) {
  if ("object" === typeof process && "function" === typeof require || "object" === typeof window || "function" === typeof importScripts) {
    throw Error("not compiled for this environment (did you build to HTML and try to run it not on the web, or set ENVIRONMENT to something - like node - and run it someplace else - like on the web?)");
  }
  "undefined" != typeof read && (Ba = function(a) {
    return read(a);
  });
  Da = function(a) {
    if ("function" === typeof readbuffer) {
      return new Uint8Array(readbuffer(a));
    }
    a = read(a, "binary");
    assert("object" === typeof a);
    return a;
  };
  Ca = function(a, b) {
    setTimeout(() => b(Da(a)), 0);
  };
  "function" === typeof quit && (wa = (a, b) => {
    Ea(b);
    quit(a);
  });
  "undefined" !== typeof print && ("undefined" === typeof console && (console = {}), console.log = print, console.warn = console.error = "undefined" !== typeof printErr ? printErr : print);
} else if (xa || ya) {
  ya ? D = self.location.href : "undefined" !== typeof document && document.currentScript && (D = document.currentScript.src);
  _scriptDir && (D = _scriptDir);
  0 !== D.indexOf("blob:") ? D = D.substr(0, D.replace(/[?#].*/, "").lastIndexOf("/") + 1) : D = "";
  if ("object" !== typeof window && "function" !== typeof importScripts) {
    throw Error("not compiled for this environment (did you build to HTML and try to run it not on the web, or set ENVIRONMENT to something - like node - and run it someplace else - like on the web?)");
  }
  x || (Ba = a => {
    var b = new XMLHttpRequest();
    b.open("GET", a, !1);
    b.send(null);
    return b.responseText;
  }, ya && (Da = a => {
    var b = new XMLHttpRequest();
    b.open("GET", a, !1);
    b.responseType = "arraybuffer";
    b.send(null);
    return new Uint8Array(b.response);
  }), Ca = (a, b, d) => {
    var e = new XMLHttpRequest();
    e.open("GET", a, !0);
    e.responseType = "arraybuffer";
    e.onload = () => {
      200 == e.status || 0 == e.status && e.response ? b(e.response) : d();
    };
    e.onerror = d;
    e.send(null);
  });
} else {
  throw Error("environment detection error");
}
x && "undefined" === typeof performance && (global.performance = require("perf_hooks").performance);
var Ka = console.log.bind(console), La = console.warn.bind(console);
x && (Ia(), Ka = a => fs.writeSync(1, a + "\n"), La = a => fs.writeSync(2, a + "\n"));
var Ma = u.print || Ka, E = u.printErr || La;
ra(u, ua);
ua = null;
Object.getOwnPropertyDescriptor(u, "arguments") || Object.defineProperty(u, "arguments", {configurable:!0, get:function() {
  v("Module.arguments has been replaced with plain arguments_ (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
u.thisProgram && (va = u.thisProgram);
Object.getOwnPropertyDescriptor(u, "thisProgram") || Object.defineProperty(u, "thisProgram", {configurable:!0, get:function() {
  v("Module.thisProgram has been replaced with plain thisProgram (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
u.quit && (wa = u.quit);
Object.getOwnPropertyDescriptor(u, "quit") || Object.defineProperty(u, "quit", {configurable:!0, get:function() {
  v("Module.quit has been replaced with plain quit_ (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
assert("undefined" === typeof u.memoryInitializerPrefixURL, "Module.memoryInitializerPrefixURL option was removed, use Module.locateFile instead");
assert("undefined" === typeof u.pthreadMainPrefixURL, "Module.pthreadMainPrefixURL option was removed, use Module.locateFile instead");
assert("undefined" === typeof u.cdInitializerPrefixURL, "Module.cdInitializerPrefixURL option was removed, use Module.locateFile instead");
assert("undefined" === typeof u.filePackagePrefixURL, "Module.filePackagePrefixURL option was removed, use Module.locateFile instead");
assert("undefined" === typeof u.read, "Module.read option was removed (modify read_ in JS)");
assert("undefined" === typeof u.readAsync, "Module.readAsync option was removed (modify readAsync in JS)");
assert("undefined" === typeof u.readBinary, "Module.readBinary option was removed (modify readBinary in JS)");
assert("undefined" === typeof u.setWindowTitle, "Module.setWindowTitle option was removed (modify setWindowTitle in JS)");
assert("undefined" === typeof u.TOTAL_MEMORY, "Module.TOTAL_MEMORY has been renamed Module.INITIAL_MEMORY");
Object.getOwnPropertyDescriptor(u, "read") || Object.defineProperty(u, "read", {configurable:!0, get:function() {
  v("Module.read has been replaced with plain read_ (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
Object.getOwnPropertyDescriptor(u, "readAsync") || Object.defineProperty(u, "readAsync", {configurable:!0, get:function() {
  v("Module.readAsync has been replaced with plain readAsync (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
Object.getOwnPropertyDescriptor(u, "readBinary") || Object.defineProperty(u, "readBinary", {configurable:!0, get:function() {
  v("Module.readBinary has been replaced with plain readBinary (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
Object.getOwnPropertyDescriptor(u, "setWindowTitle") || Object.defineProperty(u, "setWindowTitle", {configurable:!0, get:function() {
  v("Module.setWindowTitle has been replaced with plain setWindowTitle (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
assert(xa || ya || x, "Pthreads do not work in this environment yet (need Web Workers, or an alternative to them)");
assert(!za, "shell environment detected but not enabled at build time.  Add 'shell' to `-s ENVIRONMENT` to enable.");
function Na(a) {
  Oa || (Oa = {});
  Oa[a] || (Oa[a] = 1, E(a));
}
var Oa, F = 0, Pa;
u.wasmBinary && (Pa = u.wasmBinary);
Object.getOwnPropertyDescriptor(u, "wasmBinary") || Object.defineProperty(u, "wasmBinary", {configurable:!0, get:function() {
  v("Module.wasmBinary has been replaced with plain wasmBinary (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
var noExitRuntime = u.noExitRuntime || !1;
Object.getOwnPropertyDescriptor(u, "noExitRuntime") || Object.defineProperty(u, "noExitRuntime", {configurable:!0, get:function() {
  v("Module.noExitRuntime has been replaced with plain noExitRuntime (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
"object" !== typeof WebAssembly && v("no native wasm support detected");
function Qa(a, b, d = "i8") {
  "*" === d.charAt(d.length - 1) && (d = "i32");
  switch(d) {
    case "i1":
      ba()[a >> 0] = b;
      break;
    case "i8":
      ba()[a >> 0] = b;
      break;
    case "i16":
      ka()[a >> 1] = b;
      break;
    case "i32":
      t()[a >> 2] = b;
      break;
    case "i64":
      Ra = [b >>> 0, (Sa = b, 1 <= +Math.abs(Sa) ? 0 < Sa ? (Math.min(+Math.floor(Sa / 4294967296), 4294967295) | 0) >>> 0 : ~~+Math.ceil((Sa - +(~~Sa >>> 0)) / 4294967296) >>> 0 : 0)];
      t()[a >> 2] = Ra[0];
      t()[a + 4 >> 2] = Ra[1];
      break;
    case "float":
      na()[a >> 2] = b;
      break;
    case "double":
      pa()[a >> 3] = b;
      break;
    default:
      v("invalid type for setValue: " + d);
  }
}
function Ta(a, b = "i8") {
  "*" === b.charAt(b.length - 1) && (b = "i32");
  switch(b) {
    case "i1":
      return ba()[a >> 0];
    case "i8":
      return ba()[a >> 0];
    case "i16":
      return ka()[a >> 1];
    case "i32":
      return t()[a >> 2];
    case "i64":
      return t()[a >> 2];
    case "float":
      return na()[a >> 2];
    case "double":
      return Number(pa()[a >> 3]);
    default:
      v("invalid type for getValue: " + b);
  }
  return null;
}
function Ua(a) {
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
function H(a, b, d) {
  0 >= a && v("segmentation fault storing " + d + " bytes to address " + a);
  0 !== a % d && v("alignment error storing to address " + a + ", which was expected to be aligned to a multiple of " + d);
  if (Va && !Wa) {
    var e = Xa() >>> 0;
    a + d > e && v("segmentation fault, exceeded the top of the available dynamic heap when storing " + d + " bytes to address " + a + ". DYNAMICTOP=" + e);
    assert(e >= Ya());
    assert(e <= ba().length);
  }
  Qa(a, b, Ua(d));
}
function J(a, b, d) {
  0 >= a && v("segmentation fault loading " + b + " bytes from address " + a);
  0 !== a % b && v("alignment error loading from address " + a + ", which was expected to be aligned to a multiple of " + b);
  if (Va && !Wa) {
    var e = Xa() >>> 0;
    a + b > e && v("segmentation fault, exceeded the top of the available dynamic heap when loading " + b + " bytes from address " + a + ". DYNAMICTOP=" + e);
    assert(e >= Ya());
    assert(e <= ba().length);
  }
  b = Ua(b);
  a = Ta(a, b);
  d && (d = parseInt(b.substr(1), 10), a = 0 <= a ? a : 32 >= d ? 2 * Math.abs(1 << d - 1) + a : Math.pow(2, d) + a);
  return a;
}
var c, Za, $a = !1;
function assert(a, b) {
  a || v("Assertion failed" + (b ? ": " + b : ""));
}
function ab(a) {
  var b = new TextDecoder(a);
  this.decode = d => {
    assert(d instanceof Uint8Array);
    d.buffer instanceof SharedArrayBuffer && (d = new Uint8Array(d));
    return b.decode.call(b, d);
  };
}
var bb = "undefined" !== typeof TextDecoder ? new ab("utf8") : void 0;
function cb(a, b, d) {
  var e = b + d;
  for (d = b; a[d] && !(d >= e);) {
    ++d;
  }
  if (16 < d - b && a.subarray && bb) {
    return bb.decode(a.subarray(b, d));
  }
  for (e = ""; b < d;) {
    var f = a[b++];
    if (f & 128) {
      var g = a[b++] & 63;
      if (192 == (f & 224)) {
        e += String.fromCharCode((f & 31) << 6 | g);
      } else {
        var h = a[b++] & 63;
        224 == (f & 240) ? f = (f & 15) << 12 | g << 6 | h : (240 != (f & 248) && Na("Invalid UTF-8 leading byte 0x" + f.toString(16) + " encountered when deserializing a UTF-8 string in wasm memory to a JS string!"), f = (f & 7) << 18 | g << 12 | h << 6 | a[b++] & 63);
        65536 > f ? e += String.fromCharCode(f) : (f -= 65536, e += String.fromCharCode(55296 | f >> 10, 56320 | f & 1023));
      }
    } else {
      e += String.fromCharCode(f);
    }
  }
  return e;
}
function db(a, b) {
  return a ? cb(ha(), a, b) : "";
}
function eb(a, b, d, e) {
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
      b[d++] = h;
    } else {
      if (2047 >= h) {
        if (d + 1 >= e) {
          break;
        }
        b[d++] = 192 | h >> 6;
      } else {
        if (65535 >= h) {
          if (d + 2 >= e) {
            break;
          }
          b[d++] = 224 | h >> 12;
        } else {
          if (d + 3 >= e) {
            break;
          }
          1114111 < h && Na("Invalid Unicode code point 0x" + h.toString(16) + " encountered when serializing a JS string to a UTF-8 string in wasm memory! (Valid unicode code points should be in range 0-0x10FFFF).");
          b[d++] = 240 | h >> 18;
          b[d++] = 128 | h >> 12 & 63;
        }
        b[d++] = 128 | h >> 6 & 63;
      }
      b[d++] = 128 | h & 63;
    }
  }
  b[d] = 0;
  return d - f;
}
function fb(a, b, d) {
  assert("number" == typeof d, "stringToUTF8(str, outPtr, maxBytesToWrite) is missing the third parameter that specifies the length of the output buffer!");
  return eb(a, ha(), b, d);
}
function gb(a) {
  for (var b = 0, d = 0; d < a.length; ++d) {
    var e = a.charCodeAt(d);
    55296 <= e && 57343 >= e && (e = 65536 + ((e & 1023) << 10) | a.charCodeAt(++d) & 1023);
    127 >= e ? ++b : b = 2047 >= e ? b + 2 : 65535 >= e ? b + 3 : b + 4;
  }
  return b;
}
"undefined" !== typeof TextDecoder && new ab("utf-16le");
function hb(a) {
  var b = gb(a) + 1, d = ib(b);
  d && eb(a, ba(), d, b);
  return d;
}
function jb(a, b) {
  assert(0 <= a.length, "writeArrayToMemory array must have a length (should be an array or typed array)");
  ba().set(a, b);
}
var m, ea, ia, la, ma, oa, qa;
B && (m = u.buffer);
function da(a) {
  m = a;
  u.HEAP8 = ea = new Int8Array(a);
  u.HEAP16 = la = new Int16Array(a);
  u.HEAP32 = ma = new Int32Array(a);
  u.HEAPU8 = ia = new Uint8Array(a);
  u.HEAPU16 = new Uint16Array(a);
  u.HEAPU32 = new Uint32Array(a);
  u.HEAPF32 = oa = new Float32Array(a);
  u.HEAPF64 = qa = new Float64Array(a);
}
u.TOTAL_STACK && assert(5242880 === u.TOTAL_STACK, "the stack size can no longer be determined at runtime");
var kb = u.INITIAL_MEMORY || 16777216;
Object.getOwnPropertyDescriptor(u, "INITIAL_MEMORY") || Object.defineProperty(u, "INITIAL_MEMORY", {configurable:!0, get:function() {
  v("Module.INITIAL_MEMORY has been replaced with plain INITIAL_MEMORY (the initial value can be provided on Module, but after startup the value is only looked for on a local variable of that name)");
}});
assert(5242880 <= kb, "INITIAL_MEMORY should be larger than TOTAL_STACK, was " + kb + "! (TOTAL_STACK=5242880)");
assert("undefined" !== typeof Int32Array && "undefined" !== typeof Float64Array && void 0 !== Int32Array.prototype.subarray && void 0 !== Int32Array.prototype.set, "JS engine does not provide full typed array support");
if (B) {
  c = u.wasmMemory, m = u.buffer;
} else {
  if (u.wasmMemory) {
    c = u.wasmMemory;
  } else {
    if (c = new WebAssembly.Memory({initial:kb / 65536, maximum:32768, shared:!0}), !(c.buffer instanceof SharedArrayBuffer)) {
      throw E("requested a shared WebAssembly.Memory but the returned buffer is not a SharedArrayBuffer, indicating that while the browser has SharedArrayBuffer it does not have WebAssembly threads support - you may need to set a flag"), x && console.log("(on node you may need: --experimental-wasm-threads --experimental-wasm-bulk-memory and also use a recent version)"), Error("bad memory");
    }
  }
}
c && (m = c.buffer);
kb = m.byteLength;
assert(0 === kb % 65536);
da(m);
var lb;
function mb() {
  var a = nb();
  assert(0 == (a & 3));
  H(a + 4 | 0, 34821223, 4);
  H(a + 8 | 0, -1984246274, 4);
}
function ob() {
  if (!$a) {
    var a = nb(), b = J(a + 4 | 0, 4, 1) >>> 0;
    a = J(a + 8 | 0, 4, 1) >>> 0;
    34821223 == b && 2310721022 == a || v("Stack overflow! Stack cookie has been overwritten, expected hex dwords 0x89BACDFE and 0x2135467, but received 0x" + a.toString(16) + " 0x" + b.toString(16));
  }
}
var pb = new Int16Array(1), qb = new Int8Array(pb.buffer);
pb[0] = 25459;
if (115 !== qb[0] || 99 !== qb[1]) {
  throw "Runtime error: expected the system to be little-endian! (Run with -s SUPPORT_BIG_ENDIAN=1 to bypass)";
}
var rb = [], sb = [], tb = [], ub = [], Va = !1, Wa = !1, vb = 0;
function Ja() {
  return noExitRuntime || 0 < vb;
}
function wb() {
  ob();
  assert(!Va);
  Va = !0;
  B || xb(sb);
}
function yb() {
  var a = u.preRun.shift();
  rb.unshift(a);
}
assert(Math.imul, "This browser does not support Math.imul(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill");
assert(Math.fround, "This browser does not support Math.fround(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill");
assert(Math.clz32, "This browser does not support Math.clz32(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill");
assert(Math.trunc, "This browser does not support Math.trunc(), build with LEGACY_VM_SUPPORT or POLYFILL_OLD_MATH_FUNCTIONS to add in a polyfill");
var zb = 0, Ab = null, Bb = null, Cb = {};
function Db() {
  zb++;
  u.monitorRunDependencies && u.monitorRunDependencies(zb);
  assert(!Cb["wasm-instantiate"]);
  Cb["wasm-instantiate"] = 1;
  null === Ab && "undefined" !== typeof setInterval && (Ab = setInterval(function() {
    if ($a) {
      clearInterval(Ab), Ab = null;
    } else {
      var a = !1, b;
      for (b in Cb) {
        a || (a = !0, E("still waiting on run dependencies:")), E("dependency: " + b);
      }
      a && E("(end of list)");
    }
  }, 1e4));
}
u.preloadedImages = {};
u.preloadedAudios = {};
function v(a) {
  if (B) {
    postMessage({cmd:"onAbort", arg:a});
  } else {
    if (u.onAbort) {
      u.onAbort(a);
    }
  }
  a = "Aborted(" + a + ")";
  E(a);
  $a = !0;
  a = new WebAssembly.RuntimeError(a);
  ta(a);
  throw a;
}
function Eb() {
  v("Filesystem support (FS) was not included. The problem is that you are using files from JS, but files were not used from C/C++, so filesystem support was not auto-included. You can force-include filesystem support with  -s FORCE_FILESYSTEM=1");
}
u.FS_createDataFile = function() {
  Eb();
};
u.FS_createPreloadedFile = function() {
  Eb();
};
function Fb() {
  return K.startsWith("data:application/octet-stream;base64,");
}
function M(a) {
  return function() {
    var b = u.asm;
    assert(Va, "native function `" + a + "` called before runtime initialization");
    assert(!Wa, "native function `" + a + "` called after runtime exit (use NO_EXIT_RUNTIME to keep it alive after main() exits)");
    b[a] || assert(b[a], "exported native function `" + a + "` not found");
    return b[a].apply(null, arguments);
  };
}
var K;
K = "ort-wasm-simd-threaded.wasm";
Fb() || (K = Aa(K));
function Gb() {
  var a = K;
  try {
    if (a == K && Pa) {
      return new Uint8Array(Pa);
    }
    if (Da) {
      return Da(a);
    }
    throw "both async and sync fetching of the wasm failed";
  } catch (b) {
    v(b);
  }
}
function Hb() {
  if (!Pa && (xa || ya)) {
    if ("function" === typeof fetch && !K.startsWith("file://")) {
      return fetch(K, {credentials:"same-origin"}).then(function(a) {
        if (!a.ok) {
          throw "failed to load wasm binary file at '" + K + "'";
        }
        return a.arrayBuffer();
      }).catch(function() {
        return Gb();
      });
    }
    if (Ca) {
      return new Promise(function(a, b) {
        Ca(K, function(d) {
          a(new Uint8Array(d));
        }, b);
      });
    }
  }
  return Promise.resolve().then(function() {
    return Gb();
  });
}
var Sa, Ra, Ib = {};
function xb(a) {
  for (; 0 < a.length;) {
    var b = a.shift();
    if ("function" == typeof b) {
      b(u);
    } else {
      var d = b.Pa;
      "number" === typeof d ? void 0 === b.s ? N(d)() : N(d)(b.s) : d(void 0 === b.s ? null : b.s);
    }
  }
}
function Jb(a) {
  var b = O();
  a = a();
  S(b);
  return a;
}
function Kb(a) {
  assert(!B, "Internal Error! cleanupThread() can only ever be called from main application thread!");
  assert(a, "Internal Error! Null pthread_ptr in cleanupThread!");
  var b = T.i[a];
  b && (H(a | 0, 0, 4), T.Z(b.worker));
}
var T = {l:[], m:[], T:[], pa:function() {
  assert(!B);
}, qa:function() {
  T.receiveObjectTransfer = T.wa;
  T.threadInit = T.aa;
  T.setExitStatus = T.za;
}, i:{}, za:function() {
}, $:function() {
  assert(!B, "Internal Error! terminateAllThreads() can only ever be called from main application thread!");
  for (var a in T.i) {
    var b = T.i[a];
    b && b.worker && T.Z(b.worker);
  }
  assert(0 === Object.keys(T.i).length);
  assert(0 === T.m.length);
  for (a = 0; a < T.l.length; ++a) {
    b = T.l[a], assert(!b.j), b.terminate();
  }
  T.l = [];
}, Z:function(a) {
  T.ya(function() {
    delete T.i[a.j.L];
    T.l.push(a);
    T.m.splice(T.m.indexOf(a), 1);
    Lb(a.j.L);
    a.j = void 0;
  });
}, ya:function(a) {
  assert(T.ta, "runWithoutMainThreadQueuedCalls must be done on the main runtime thread");
  assert(Mb);
  t()[Mb >> 2] = 0;
  try {
    a();
  } finally {
    t()[Mb >> 2] = 1;
  }
}, wa:function() {
}, aa:function() {
  for (var a in T.T) {
    T.T[a]();
  }
}, sa:function(a, b) {
  a.onmessage = d => {
    d = d.data;
    var e = d.cmd;
    a.j && (T.ga = a.j.L);
    if (d.targetThread && d.targetThread != Nb()) {
      var f = T.i[d.Xa];
      f ? f.worker.postMessage(d, d.transferList) : E('Internal error! Worker sent a message "' + e + '" to target pthread ' + d.targetThread + ", but that thread no longer exists!");
    } else {
      if ("processQueuedMainThreadWork" === e) {
        Ob();
      } else if ("spawnThread" === e) {
        Pb(d);
      } else if ("cleanupThread" === e) {
        Kb(d.thread);
      } else if ("killThread" === e) {
        d = d.thread, assert(!B, "Internal Error! killThread() can only ever be called from main application thread!"), assert(d, "Internal Error! Null pthread_ptr in killThread!"), H(d | 0, 0, 4), e = T.i[d], delete T.i[d], e.worker.terminate(), Lb(d), T.m.splice(T.m.indexOf(e.worker), 1), e.worker.j = void 0;
      } else if ("cancelThread" === e) {
        d = d.thread, assert(!B, "Internal Error! cancelThread() can only ever be called from main application thread!"), assert(d, "Internal Error! Null pthread_ptr in cancelThread!"), T.i[d].worker.postMessage({cmd:"cancel"});
      } else if ("loaded" === e) {
        a.loaded = !0, b && b(a), a.A && (a.A(), delete a.A);
      } else if ("print" === e) {
        Ma("Thread " + d.threadId + ": " + d.text);
      } else if ("printErr" === e) {
        E("Thread " + d.threadId + ": " + d.text);
      } else if ("alert" === e) {
        alert("Thread " + d.threadId + ": " + d.text);
      } else if ("setimmediate" === d.target) {
        a.postMessage(d);
      } else if ("onAbort" === e) {
        if (u.onAbort) {
          u.onAbort(d.arg);
        }
      } else {
        E("worker sent an unknown command " + e);
      }
    }
    T.ga = void 0;
  };
  a.onerror = d => {
    var e = "worker sent an error!";
    if (a.j) {
      var f = a.j.L;
      f && (e = "Pthread 0x" + f.toString(16) + " sent an error!");
    }
    E(e + " " + d.filename + ":" + d.lineno + ": " + d.message);
    throw d;
  };
  x && (a.on("message", function(d) {
    a.onmessage({data:d});
  }), a.on("error", function(d) {
    a.onerror(d);
  }), a.on("detachedExit", function() {
  }));
  assert(c instanceof WebAssembly.Memory, "WebAssembly memory should have been loaded by now!");
  assert(Za instanceof WebAssembly.Module, "WebAssembly Module should have been loaded by now!");
  a.postMessage({cmd:"load", urlOrBlob:u.mainScriptUrlOrBlob || _scriptDir, wasmMemory:c, wasmModule:Za});
}, da:function() {
  var a = Aa("ort-wasm-simd-threaded.worker.js");
  T.l.push(new Worker(a));
}, ia:function() {
  0 == T.l.length && (E("Tried to spawn a new thread, but the thread pool is exhausted.\nThis might result in a deadlock unless some threads eventually exit or the code explicitly breaks out to the event loop.\nIf you want to increase the pool size, use setting `-s PTHREAD_POOL_SIZE=...`.\nIf you want to throw an explicit error instead of the risk of deadlocking in those cases, use setting `-s PTHREAD_POOL_SIZE_STRICT=2`."), T.da(), T.sa(T.l[0]));
  return T.l.pop();
}};
u.establishStackSpace = function() {
  var a = Nb(), b = J(a + 44 | 0, 4, 0) | 0;
  a = J(a + 48 | 0, 4, 0) | 0;
  a = b - a;
  assert(0 != b);
  assert(0 != a);
  assert(b > a);
  Qb(b, a);
  S(b);
  mb();
};
function Rb(a) {
  if (B) {
    return W(1, 0, a);
  }
  try {
    Sb(a);
  } catch (b) {
    b instanceof Fa || "unwind" == b || wa(1, b);
  }
}
var Tb = [];
function N(a) {
  var b = Tb[a];
  b || (a >= Tb.length && (Tb.length = a + 1), Tb[a] = b = lb.get(a));
  assert(lb.get(a) == b, "JavaScript-side Wasm function table mirror is out of date!");
  return b;
}
u.invokeEntryPoint = function(a, b) {
  return N(a)(b);
};
var Ub;
Ub = x ? () => {
  var a = process.hrtime();
  return 1e3 * a[0] + a[1] / 1e6;
} : B ? () => performance.now() - u.__performance_now_clock_drift : () => performance.now();
function Vb(a, b) {
  if (0 === a) {
    a = Date.now();
  } else if (1 === a || 4 === a) {
    a = Ub();
  } else {
    return H(Wb() | 0, 28, 4), -1;
  }
  H(b | 0, a / 1e3 | 0, 4);
  H(b + 4 | 0, a % 1e3 * 1E6 | 0, 4);
  return 0;
}
function Xb(a) {
  this.V = a;
  this.g = a - 16;
  this.Ca = function(b) {
    H(this.g + 4 | 0, b | 0, 4);
  };
  this.v = function() {
    return J(this.g + 4 | 0, 4, 0) | 0;
  };
  this.Aa = function(b) {
    H(this.g + 8 | 0, b | 0, 4);
  };
  this.la = function() {
    return J(this.g + 8 | 0, 4, 0) | 0;
  };
  this.Ba = function() {
    H(this.g | 0, 0, 4);
  };
  this.R = function(b) {
    H(this.g + 12 | 0, (b ? 1 : 0) | 0, 1);
  };
  this.ka = function() {
    return 0 != (J(this.g + 12 | 0, 1, 0) | 0);
  };
  this.S = function(b) {
    H(this.g + 13 | 0, (b ? 1 : 0) | 0, 1);
  };
  this.X = function() {
    return 0 != (J(this.g + 13 | 0, 1, 0) | 0);
  };
  this.na = function(b, d) {
    this.Ca(b);
    this.Aa(d);
    this.Ba();
    this.R(!1);
    this.S(!1);
  };
  this.ba = function() {
    Atomics.add(t(), this.g + 0 >> 2, 1);
  };
  this.xa = function() {
    var b = Atomics.sub(t(), this.g + 0 >> 2, 1);
    assert(0 < b);
    return 1 === b;
  };
}
function Yb(a) {
  this.P = function() {
    Zb(this.g);
    this.g = 0;
  };
  this.K = function(b) {
    H(this.g | 0, b | 0, 4);
  };
  this.u = function() {
    return J(this.g | 0, 4, 0) | 0;
  };
  this.B = function(b) {
    H(this.g + 4 | 0, b | 0, 4);
  };
  this.H = function() {
    return this.g + 4;
  };
  this.ja = function() {
    return J(this.g + 4 | 0, 4, 0) | 0;
  };
  this.ma = function() {
    if ($b(this.I().v())) {
      return J(this.u() | 0, 4, 0) | 0;
    }
    var b = this.ja();
    return 0 !== b ? b : this.u();
  };
  this.I = function() {
    return new Xb(this.u());
  };
  void 0 === a ? (this.g = ib(8), this.B(0)) : this.g = a;
}
var ac = [], bc = 0, cc = 0;
function dc(a) {
  try {
    return Zb((new Xb(a)).g);
  } catch (b) {
    E("exception during cxa_free_exception: " + b);
  }
}
function Pb(a) {
  assert(!B, "Internal Error! spawnThread() can only ever be called from main application thread!");
  assert(a.J, "Internal error, no pthread ptr!");
  var b = T.ia();
  if (!b) {
    return 6;
  }
  assert(!b.j, "Internal error!");
  T.m.push(b);
  var d = T.i[a.J] = {worker:b, L:a.J};
  b.j = d;
  var e = {cmd:"run", start_routine:a.Da, arg:a.s, threadInfoStruct:a.J};
  b.A = () => {
    e.time = performance.now();
    b.postMessage(e, a.Ia);
  };
  b.loaded && (b.A(), delete b.A);
  return 0;
}
var ec = {}, fc = [null, [], []];
function gc(a, b) {
  var d = fc[a];
  assert(d);
  0 === b || 10 === b ? ((1 === a ? Ma : E)(cb(d, 0)), d.length = 0) : d.push(b);
}
var hc = {};
function ic(a, b, d) {
  return B ? W(2, 1, a, b, d) : 0;
}
function jc(a, b) {
  if (B) {
    return W(3, 1, a, b);
  }
  v("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}
function kc(a, b, d, e) {
  if (B) {
    return W(4, 1, a, b, d, e);
  }
  v("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}
function lc(a, b) {
  if (B) {
    return W(5, 1, a, b);
  }
  v("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}
function mc(a, b, d) {
  if (B) {
    return W(6, 1, a, b, d);
  }
  v("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}
function nc(a, b, d) {
  return B ? W(7, 1, a, b, d) : 0;
}
function oc(a, b) {
  if (B) {
    return W(8, 1, a, b);
  }
  v("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}
function pc(a, b) {
  if (B) {
    return W(9, 1, a, b);
  }
  a = db(a);
  return hc.Ma(a, b);
}
function qc(a, b, d, e, f, g) {
  if (B) {
    b = W(10, 1, a, b, d, e, f, g);
  } else {
    if (g <<= 12, 0 !== (e & 16) && 0 !== a % 65536) {
      b = -28;
    } else {
      if (0 !== (e & 32)) {
        a = b;
        assert(65536, "alignment argument is required");
        var h = 65536 * Math.ceil(a / 65536);
        (a = rc(65536, h)) ? ha().fill(0, a, a + h) : a = 0;
        a ? (ec[a] = {va:a, ra:b, ea:!0, fd:f, Ua:d, flags:e, offset:g}, b = a) : b = -48;
      } else {
        b = -52;
      }
    }
  }
  return b;
}
function sc(a, b) {
  if (B) {
    a = W(11, 1, a, b);
  } else {
    var d = ec[a];
    0 !== b && d ? (b === d.ra && (assert(ec[a].flags & 32), ec[a] = null, d.ea && Zb(d.va)), a = 0) : a = -28;
  }
  return a;
}
function tc(a, b, d) {
  if (B) {
    return W(12, 1, a, b, d);
  }
  v("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}
function uc(a, b, d) {
  if (B) {
    return W(13, 1, a, b, d);
  }
  a = db(a);
  return hc.Na(a, b, d);
}
function vc(a) {
  if (B) {
    return W(14, 1, a);
  }
  v("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}
function wc(a, b) {
  if (B) {
    return W(15, 1, a, b);
  }
  v("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}
function xc(a) {
  if (B) {
    return W(16, 1, a);
  }
  v("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}
function yc(a, b, d) {
  function e(l) {
    return (l = l.toTimeString().match(/\(([A-Za-z ]+)\)$/)) ? l[1] : "GMT";
  }
  if (B) {
    return W(17, 1, a, b, d);
  }
  var f = (new Date()).getFullYear(), g = new Date(f, 0, 1), h = new Date(f, 6, 1);
  f = g.getTimezoneOffset();
  var k = h.getTimezoneOffset();
  H(a | 0, 60 * Math.max(f, k) | 0, 4);
  H(b | 0, Number(f != k) | 0, 4);
  a = e(g);
  b = e(h);
  a = hb(a);
  b = hb(b);
  k < f ? (H(d | 0, a | 0, 4), H(d + 4 | 0, b | 0, 4)) : (H(d | 0, b | 0, 4), H(d + 4 | 0, a | 0, 4));
}
function zc(a, b, d) {
  zc.fa || (zc.fa = !0, yc(a, b, d));
}
function W(a, b) {
  var d = arguments.length - 2, e = arguments;
  if (19 < d) {
    throw "emscripten_proxy_to_main_thread_js: Too many arguments " + d + " to proxied function idx=" + a + ", maximum supported is 19!";
  }
  return Jb(function() {
    for (var f = Ac(8 * d), g = f >> 3, h = 0; h < d; h++) {
      var k = e[2 + h];
      pa()[g + h] = k;
    }
    return Bc(a, d, f, b);
  });
}
var Cc = [];
function Dc(a, b, d, e) {
  Jb(function() {
    var f = Ac(12), g = 0;
    if (b) {
      g = gb(b) + 1;
      var h = ib(g);
      fb(b, h, g);
      g = h;
    }
    H(f | 0, g | 0, 4);
    H(f + 4 | 0, d | 0, 4);
    H(f + 8 | 0, e | 0, 4);
    Ec(a, 657457152, 0, g, f);
  });
}
var Fc = [0, "undefined" !== typeof document ? document : 0, "undefined" !== typeof window ? window : 0];
function Gc(a) {
  a = 2 < a ? db(a) : a;
  return Fc[a] || ("undefined" !== typeof document ? document.querySelector(a) : void 0);
}
function Hc(a, b, d) {
  var e = Gc(a);
  if (!e) {
    return -4;
  }
  e.G && (H(e.G | 0, b | 0, 4), H(e.G + 4 | 0, d | 0, 4));
  if (e.Y || !e.Ka) {
    e.Y && (e = e.Y), a = !1, e.F && e.F.D && (a = e.F.D.getParameter(2978), a = 0 === a[0] && 0 === a[1] && a[2] === e.width && a[3] === e.height), e.width = b, e.height = d, a && e.F.D.viewport(0, 0, b, d);
  } else {
    return e.G ? (e = J(e.G + 8 | 0, 4, 0) | 0, a = a ? db(a) : "", Dc(e, a, b, d), 1) : -4;
  }
  return 0;
}
function Ic(a, b, d) {
  return B ? W(18, 1, a, b, d) : Hc(a, b, d);
}
function Jc(a) {
  var b = a.getExtension("ANGLE_instanced_arrays");
  b && (a.vertexAttribDivisor = function(d, e) {
    b.vertexAttribDivisorANGLE(d, e);
  }, a.drawArraysInstanced = function(d, e, f, g) {
    b.drawArraysInstancedANGLE(d, e, f, g);
  }, a.drawElementsInstanced = function(d, e, f, g, h) {
    b.drawElementsInstancedANGLE(d, e, f, g, h);
  });
}
function Kc(a) {
  var b = a.getExtension("OES_vertex_array_object");
  b && (a.createVertexArray = function() {
    return b.createVertexArrayOES();
  }, a.deleteVertexArray = function(d) {
    b.deleteVertexArrayOES(d);
  }, a.bindVertexArray = function(d) {
    b.bindVertexArrayOES(d);
  }, a.isVertexArray = function(d) {
    return b.isVertexArrayOES(d);
  });
}
function Lc(a) {
  var b = a.getExtension("WEBGL_draw_buffers");
  b && (a.drawBuffers = function(d, e) {
    b.drawBuffersWEBGL(d, e);
  });
}
function Mc(a, b) {
  a.W || (a.W = a.getContext, a.getContext = function(e, f) {
    f = a.W(e, f);
    return "webgl" == e == f instanceof WebGLRenderingContext ? f : null;
  });
  var d = a.getContext("webgl", b);
  return d ? Nc(d, b) : 0;
}
function Nc(a, b) {
  var d = ib(8);
  H(d + 4 | 0, Nb() | 0, 4);
  var e = {Ra:d, attributes:b, version:b.ua, D:a};
  a.canvas && (a.canvas.F = e);
  ("undefined" === typeof b.U || b.U) && Oc(e);
  return d;
}
function Oc(a) {
  a || (a = Pc);
  if (!a.oa) {
    a.oa = !0;
    var b = a.D;
    Jc(b);
    Kc(b);
    Lc(b);
    b.La = b.getExtension("EXT_disjoint_timer_query");
    b.Ta = b.getExtension("WEBGL_multi_draw");
    (b.getSupportedExtensions() || []).forEach(function(d) {
      d.includes("lose_context") || d.includes("debug") || b.getExtension(d);
    });
  }
}
var Pc, Qc = ["default", "low-power", "high-performance"], Rc = {};
function Sc() {
  if (!Tc) {
    var a = {USER:"web_user", LOGNAME:"web_user", PATH:"/", PWD:"/", HOME:"/home/web_user", LANG:("object" === typeof navigator && navigator.languages && navigator.languages[0] || "C").replace("-", "_") + ".UTF-8", _:va || "./this.program"}, b;
    for (b in Rc) {
      void 0 === Rc[b] ? delete a[b] : a[b] = Rc[b];
    }
    var d = [];
    for (b in a) {
      d.push(b + "=" + a[b]);
    }
    Tc = d;
  }
  return Tc;
}
var Tc;
function Uc(a, b) {
  if (B) {
    return W(19, 1, a, b);
  }
  var d = 0;
  Sc().forEach(function(e, f) {
    var g = b + d;
    H(a + 4 * f | 0, g | 0, 4);
    f = g;
    for (g = 0; g < e.length; ++g) {
      assert(e.charCodeAt(g) === (e.charCodeAt(g) & 255)), H(f++ | 0, e.charCodeAt(g) | 0, 1);
    }
    H(f | 0, 0, 1);
    d += e.length + 1;
  });
  return 0;
}
function Vc(a, b) {
  if (B) {
    return W(20, 1, a, b);
  }
  var d = Sc();
  H(a | 0, d.length | 0, 4);
  var e = 0;
  d.forEach(function(f) {
    e += f.length + 1;
  });
  H(b | 0, e | 0, 4);
  return 0;
}
function Wc(a) {
  if (B) {
    return W(21, 1, a);
  }
  v("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
  return 0;
}
function Xc(a, b, d, e) {
  if (B) {
    return W(22, 1, a, b, d, e);
  }
  a = hc.Qa(a);
  b = hc.Oa(a, b, d);
  H(e | 0, b | 0, 4);
  return 0;
}
function Yc(a, b, d, e, f) {
  if (B) {
    return W(23, 1, a, b, d, e, f);
  }
  v("it should not be possible to operate on streams when !SYSCALLS_REQUIRE_FILESYSTEM");
}
function Zc(a, b, d, e) {
  if (B) {
    return W(24, 1, a, b, d, e);
  }
  for (var f = 0, g = 0; g < d; g++) {
    var h = J(b | 0, 4, 0) | 0, k = J(b + 4 | 0, 4, 0) | 0;
    b += 8;
    for (var l = 0; l < k; l++) {
      gc(a, ha()[h + l]);
    }
    f += k;
  }
  H(e | 0, f | 0, 4);
  return 0;
}
function $c(a) {
  return 0 === a % 4 && (0 !== a % 100 || 0 === a % 400);
}
function ad(a, b) {
  for (var d = 0, e = 0; e <= b; d += a[e++]) {
  }
  return d;
}
var bd = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], cd = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
function dd(a, b) {
  for (a = new Date(a.getTime()); 0 < b;) {
    var d = a.getMonth(), e = ($c(a.getFullYear()) ? bd : cd)[d];
    if (b > e - a.getDate()) {
      b -= e - a.getDate() + 1, a.setDate(1), 11 > d ? a.setMonth(d + 1) : (a.setMonth(0), a.setFullYear(a.getFullYear() + 1));
    } else {
      a.setDate(a.getDate() + b);
      break;
    }
  }
  return a;
}
function ed(a, b, d, e) {
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
    p = dd(new Date(p.h + 1900, 0, 1), p.O);
    var w = new Date(p.getFullYear() + 1, 0, 4), z = k(new Date(p.getFullYear(), 0, 4));
    w = k(w);
    return 0 >= h(z, p) ? 0 >= h(w, p) ? p.getFullYear() + 1 : p.getFullYear() : p.getFullYear() - 1;
  }
  var n = J(e + 40 | 0, 4, 0) | 0;
  e = {Ga:J(e | 0, 4, 0) | 0, Fa:J(e + 4 | 0, 4, 0) | 0, M:J(e + 8 | 0, 4, 0) | 0, C:J(e + 12 | 0, 4, 0) | 0, o:J(e + 16 | 0, 4, 0) | 0, h:J(e + 20 | 0, 4, 0) | 0, N:J(e + 24 | 0, 4, 0) | 0, O:J(e + 28 | 0, 4, 0) | 0, Ya:J(e + 32 | 0, 4, 0) | 0, Ea:J(e + 36 | 0, 4, 0) | 0, Ha:n ? db(n) : ""};
  d = db(d);
  n = {"%c":"%a %b %d %H:%M:%S %Y", "%D":"%m/%d/%y", "%F":"%Y-%m-%d", "%h":"%b", "%r":"%I:%M:%S %p", "%R":"%H:%M", "%T":"%H:%M:%S", "%x":"%m/%d/%y", "%X":"%H:%M:%S", "%Ec":"%c", "%EC":"%C", "%Ex":"%m/%d/%y", "%EX":"%H:%M:%S", "%Ey":"%y", "%EY":"%Y", "%Od":"%d", "%Oe":"%e", "%OH":"%H", "%OI":"%I", "%Om":"%m", "%OM":"%M", "%OS":"%S", "%Ou":"%u", "%OU":"%U", "%OV":"%V", "%Ow":"%w", "%OW":"%W", "%Oy":"%y"};
  for (var q in n) {
    d = d.replace(new RegExp(q, "g"), n[q]);
  }
  var r = "Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "), y = "January February March April May June July August September October November December".split(" ");
  n = {"%a":function(p) {
    return r[p.N].substring(0, 3);
  }, "%A":function(p) {
    return r[p.N];
  }, "%b":function(p) {
    return y[p.o].substring(0, 3);
  }, "%B":function(p) {
    return y[p.o];
  }, "%C":function(p) {
    return g((p.h + 1900) / 100 | 0, 2);
  }, "%d":function(p) {
    return g(p.C, 2);
  }, "%e":function(p) {
    return f(p.C, 2, " ");
  }, "%g":function(p) {
    return l(p).toString().substring(2);
  }, "%G":function(p) {
    return l(p);
  }, "%H":function(p) {
    return g(p.M, 2);
  }, "%I":function(p) {
    p = p.M;
    0 == p ? p = 12 : 12 < p && (p -= 12);
    return g(p, 2);
  }, "%j":function(p) {
    return g(p.C + ad($c(p.h + 1900) ? bd : cd, p.o - 1), 3);
  }, "%m":function(p) {
    return g(p.o + 1, 2);
  }, "%M":function(p) {
    return g(p.Fa, 2);
  }, "%n":function() {
    return "\n";
  }, "%p":function(p) {
    return 0 <= p.M && 12 > p.M ? "AM" : "PM";
  }, "%S":function(p) {
    return g(p.Ga, 2);
  }, "%t":function() {
    return "\t";
  }, "%u":function(p) {
    return p.N || 7;
  }, "%U":function(p) {
    var w = new Date(p.h + 1900, 0, 1), z = 0 === w.getDay() ? w : dd(w, 7 - w.getDay());
    p = new Date(p.h + 1900, p.o, p.C);
    return 0 > h(z, p) ? g(Math.ceil((31 - z.getDate() + (ad($c(p.getFullYear()) ? bd : cd, p.getMonth() - 1) - 31) + p.getDate()) / 7), 2) : 0 === h(z, w) ? "01" : "00";
  }, "%V":function(p) {
    var w = new Date(p.h + 1901, 0, 4), z = k(new Date(p.h + 1900, 0, 4));
    w = k(w);
    var A = dd(new Date(p.h + 1900, 0, 1), p.O);
    return 0 > h(A, z) ? "53" : 0 >= h(w, A) ? "01" : g(Math.ceil((z.getFullYear() < p.h + 1900 ? p.O + 32 - z.getDate() : p.O + 1 - z.getDate()) / 7), 2);
  }, "%w":function(p) {
    return p.N;
  }, "%W":function(p) {
    var w = new Date(p.h, 0, 1), z = 1 === w.getDay() ? w : dd(w, 0 === w.getDay() ? 1 : 7 - w.getDay() + 1);
    p = new Date(p.h + 1900, p.o, p.C);
    return 0 > h(z, p) ? g(Math.ceil((31 - z.getDate() + (ad($c(p.getFullYear()) ? bd : cd, p.getMonth() - 1) - 31) + p.getDate()) / 7), 2) : 0 === h(z, w) ? "01" : "00";
  }, "%y":function(p) {
    return (p.h + 1900).toString().substring(2);
  }, "%Y":function(p) {
    return p.h + 1900;
  }, "%z":function(p) {
    p = p.Ea;
    var w = 0 <= p;
    p = Math.abs(p) / 60;
    return (w ? "+" : "-") + String("0000" + (p / 60 * 100 + p % 60)).slice(-4);
  }, "%Z":function(p) {
    return p.Ha;
  }, "%%":function() {
    return "%";
  }};
  for (q in n) {
    d.includes(q) && (d = d.replace(new RegExp(q, "g"), n[q](e)));
  }
  q = fd(d);
  if (q.length > b) {
    return 0;
  }
  jb(q, a);
  return q.length - 1;
}
B || T.pa();
var gd = [null, Rb, ic, jc, kc, lc, mc, nc, oc, pc, qc, sc, tc, uc, vc, wc, xc, yc, Ic, Uc, Vc, Wc, Xc, Yc, Zc];
function fd(a) {
  var b = Array(gb(a) + 1);
  eb(a, b, 0, b.length);
  return b;
}
var ih = {__assert_fail:function(a, b, d, e) {
  v("Assertion failed: " + db(a) + ", at: " + [b ? db(b) : "unknown filename", d, e ? db(e) : "unknown function"]);
}, __clock_gettime:function(a, b) {
  return Vb(a, b);
}, __cxa_allocate_exception:function(a) {
  return ib(a + 16) + 16;
}, __cxa_begin_catch:function(a) {
  a = new Yb(a);
  var b = a.I();
  b.ka() || (b.R(!0), bc--);
  b.S(!1);
  ac.push(a);
  b.ba();
  return a.ma();
}, __cxa_call_unexpected:function(a) {
  E("Unexpected exception thrown, this is not properly supported - aborting");
  $a = !0;
  throw a;
}, __cxa_end_catch:function() {
  X(0);
  assert(0 < ac.length);
  var a = ac.pop(), b = a.I();
  if (b.xa() && !b.X()) {
    var d = b.la();
    d && N(d)(b.V);
    dc(b.V);
  }
  a.P();
  cc = 0;
}, __cxa_find_matching_catch_2:function() {
  var a = cc;
  if (!a) {
    return F = 0;
  }
  var b = (new Xb(a)).v(), d = new Yb();
  d.K(a);
  d.B(a);
  if (!b) {
    return F = 0, d.g | 0;
  }
  a = Array.prototype.slice.call(arguments);
  for (var e = 0; e < a.length; e++) {
    var f = a[e];
    if (0 === f || f === b) {
      break;
    }
    if (hd(f, b, d.H())) {
      return F = f, d.g | 0;
    }
  }
  F = b;
  return d.g | 0;
}, __cxa_find_matching_catch_3:function() {
  var a = cc;
  if (!a) {
    return F = 0;
  }
  var b = (new Xb(a)).v(), d = new Yb();
  d.K(a);
  d.B(a);
  if (!b) {
    return F = 0, d.g | 0;
  }
  a = Array.prototype.slice.call(arguments);
  for (var e = 0; e < a.length; e++) {
    var f = a[e];
    if (0 === f || f === b) {
      break;
    }
    if (hd(f, b, d.H())) {
      return F = f, d.g | 0;
    }
  }
  F = b;
  return d.g | 0;
}, __cxa_find_matching_catch_4:function() {
  var a = cc;
  if (!a) {
    return F = 0;
  }
  var b = (new Xb(a)).v(), d = new Yb();
  d.K(a);
  d.B(a);
  if (!b) {
    return F = 0, d.g | 0;
  }
  a = Array.prototype.slice.call(arguments);
  for (var e = 0; e < a.length; e++) {
    var f = a[e];
    if (0 === f || f === b) {
      break;
    }
    if (hd(f, b, d.H())) {
      return F = f, d.g | 0;
    }
  }
  F = b;
  return d.g | 0;
}, __cxa_find_matching_catch_5:function() {
  var a = cc;
  if (!a) {
    return F = 0;
  }
  var b = (new Xb(a)).v(), d = new Yb();
  d.K(a);
  d.B(a);
  if (!b) {
    return F = 0, d.g | 0;
  }
  a = Array.prototype.slice.call(arguments);
  for (var e = 0; e < a.length; e++) {
    var f = a[e];
    if (0 === f || f === b) {
      break;
    }
    if (hd(f, b, d.H())) {
      return F = f, d.g | 0;
    }
  }
  F = b;
  return d.g | 0;
}, __cxa_free_exception:dc, __cxa_rethrow:function() {
  var a = ac.pop();
  a || v("no exception to throw");
  var b = a.I(), d = a.u();
  b.X() ? a.P() : (ac.push(a), b.S(!0), b.R(!1), bc++);
  cc = d;
  throw d;
}, __cxa_throw:function(a, b, d) {
  (new Xb(a)).na(b, d);
  cc = a;
  bc++;
  throw a;
}, __cxa_uncaught_exceptions:function() {
  return bc;
}, __emscripten_init_main_thread_js:function(a) {
  jd(a, !ya, 1, !xa);
  T.ta = !0;
  assert(0 < kd);
  T.aa();
}, __emscripten_thread_cleanup:function(a) {
  B ? postMessage({cmd:"cleanupThread", thread:a}) : Kb(a);
}, __pthread_create_js:function(a, b, d, e) {
  if ("undefined" === typeof SharedArrayBuffer) {
    return E("Current environment does not support SharedArrayBuffer, pthreads are not available!"), 6;
  }
  var f = [];
  if (B && 0 === f.length) {
    return ld(687865856, a, b, d, e);
  }
  a = {Da:d, J:a, s:e, Ia:f};
  return B ? (a.Ja = "spawnThread", postMessage(a, f), 0) : Pb(a);
}, __resumeException:function(a) {
  a = new Yb(a);
  var b = a.u();
  cc || (cc = b);
  a.P();
  throw b;
}, __syscall_fcntl64:ic, __syscall_fstat64:jc, __syscall_fstatat64:kc, __syscall_getcwd:lc, __syscall_getdents64:mc, __syscall_ioctl:nc, __syscall_lstat64:oc, __syscall_mkdir:pc, __syscall_mmap2:qc, __syscall_munmap:sc, __syscall_open:tc, __syscall_readlink:uc, __syscall_rmdir:vc, __syscall_stat64:wc, __syscall_unlink:xc, _dlopen_js:function() {
  v("To use dlopen, you need to use Emscripten's linking support, see https://github.com/emscripten-core/emscripten/wiki/Linking");
}, _dlsym_js:function() {
  v("To use dlopen, you need to use Emscripten's linking support, see https://github.com/emscripten-core/emscripten/wiki/Linking");
}, _emscripten_default_pthread_stack_size:function() {
  return 2097152;
}, _emscripten_futex_wait_non_blocking:function(a, b, d) {
  assert(xa);
  var e = performance.now();
  d = e + d;
  assert(0 < kd);
  e = Atomics.exchange(t(), kd >> 2, a);
  for (assert(0 == e);;) {
    e = performance.now();
    if (e > d) {
      return e = Atomics.exchange(t(), kd >> 2, 0), assert(e == a || 0 == e), -73;
    }
    e = Atomics.exchange(t(), kd >> 2, 0);
    assert(e == a || 0 == e);
    if (0 == e) {
      break;
    }
    Ob();
    if (Atomics.load(t(), a >> 2) != b) {
      return -6;
    }
    e = Atomics.exchange(t(), kd >> 2, a);
    assert(0 == e);
  }
  return 0;
}, _emscripten_notify_thread_queue:function(a, b) {
  if (a == b) {
    postMessage({cmd:"processQueuedMainThreadWork"});
  } else if (B) {
    postMessage({targetThread:a, cmd:"processThreadQueue"});
  } else {
    b = (b = T.i[a]) && b.worker;
    if (!b) {
      E("Cannot send message to thread with ID " + a + ", unknown thread ID!");
      return;
    }
    b.postMessage({cmd:"processThreadQueue"});
  }
  return 1;
}, _gmtime_js:function(a, b) {
  a = new Date(1e3 * (J(a | 0, 4, 0) | 0));
  H(b | 0, a.getUTCSeconds() | 0, 4);
  H(b + 4 | 0, a.getUTCMinutes() | 0, 4);
  H(b + 8 | 0, a.getUTCHours() | 0, 4);
  H(b + 12 | 0, a.getUTCDate() | 0, 4);
  H(b + 16 | 0, a.getUTCMonth() | 0, 4);
  H(b + 20 | 0, a.getUTCFullYear() - 1900 | 0, 4);
  H(b + 24 | 0, a.getUTCDay() | 0, 4);
  H(b + 28 | 0, (a.getTime() - Date.UTC(a.getUTCFullYear(), 0, 1, 0, 0, 0, 0)) / 864E5 | 0, 4);
}, _localtime_js:function(a, b) {
  a = new Date(1e3 * (J(a | 0, 4, 0) | 0));
  H(b | 0, a.getSeconds() | 0, 4);
  H(b + 4 | 0, a.getMinutes() | 0, 4);
  H(b + 8 | 0, a.getHours() | 0, 4);
  H(b + 12 | 0, a.getDate() | 0, 4);
  H(b + 16 | 0, a.getMonth() | 0, 4);
  H(b + 20 | 0, a.getFullYear() - 1900 | 0, 4);
  H(b + 24 | 0, a.getDay() | 0, 4);
  var d = new Date(a.getFullYear(), 0, 1);
  H(b + 28 | 0, (a.getTime() - d.getTime()) / 864E5 | 0, 4);
  H(b + 36 | 0, -(60 * a.getTimezoneOffset()) | 0, 4);
  var e = (new Date(a.getFullYear(), 6, 1)).getTimezoneOffset();
  d = d.getTimezoneOffset();
  H(b + 32 | 0, (e != d && a.getTimezoneOffset() == Math.min(d, e)) | 0, 4);
}, _mktime_js:function(a) {
  var b = new Date((J(a + 20 | 0, 4, 0) | 0) + 1900, J(a + 16 | 0, 4, 0) | 0, J(a + 12 | 0, 4, 0) | 0, J(a + 8 | 0, 4, 0) | 0, J(a + 4 | 0, 4, 0) | 0, J(a | 0, 4, 0) | 0, 0), d = J(a + 32 | 0, 4, 0) | 0, e = b.getTimezoneOffset(), f = new Date(b.getFullYear(), 0, 1), g = (new Date(b.getFullYear(), 6, 1)).getTimezoneOffset(), h = f.getTimezoneOffset(), k = Math.min(h, g);
  0 > d ? H(a + 32 | 0, Number(g != h && k == e) | 0, 4) : 0 < d != (k == e) && (g = Math.max(h, g), b.setTime(b.getTime() + 6e4 * ((0 < d ? k : g) - e)));
  H(a + 24 | 0, b.getDay() | 0, 4);
  H(a + 28 | 0, (b.getTime() - f.getTime()) / 864E5 | 0, 4);
  H(a | 0, b.getSeconds() | 0, 4);
  H(a + 4 | 0, b.getMinutes() | 0, 4);
  H(a + 8 | 0, b.getHours() | 0, 4);
  H(a + 12 | 0, b.getDate() | 0, 4);
  H(a + 16 | 0, b.getMonth() | 0, 4);
  return b.getTime() / 1e3 | 0;
}, _tzset_js:zc, abort:function() {
  v("native code called abort()");
}, alignfault:function() {
  v("alignment fault");
}, clock_gettime:Vb, difftime:function(a, b) {
  return a - b;
}, emscripten_check_blocking_allowed:function() {
  x || ya || Na("Blocking on the main thread is very dangerous, see https://emscripten.org/docs/porting/pthreads.html#blocking-on-the-main-browser-thread");
}, emscripten_get_heap_max:function() {
  return 2147483648;
}, emscripten_get_now:Ub, emscripten_memcpy_big:function(a, b, d) {
  ha().copyWithin(a, b, b + d);
}, emscripten_num_logical_cores:function() {
  return x ? require("os").cpus().length : navigator.hardwareConcurrency;
}, emscripten_receive_on_main_thread_js:function(a, b, d) {
  Cc.length = b;
  d >>= 3;
  for (var e = 0; e < b; e++) {
    Cc[e] = pa()[d + e];
  }
  a = 0 > a ? Ib[-a - 1] : gd[a];
  assert(a.length == b, "Call args mismatch in emscripten_receive_on_main_thread_js");
  return a.apply(null, Cc);
}, emscripten_resize_heap:function(a) {
  var b = ha().length;
  a >>>= 0;
  if (a <= b) {
    return !1;
  }
  if (2147483648 < a) {
    return E("Cannot enlarge memory, asked to go up to " + a + " bytes, but the limit is 2147483648 bytes!"), !1;
  }
  for (var d = 1; 4 >= d; d *= 2) {
    var e = b * (1 + .2 / d);
    e = Math.min(e, a + 100663296);
    e = Math.max(a, e);
    0 < e % 65536 && (e += 65536 - e % 65536);
    e = Math.min(2147483648, e);
    var f = Ub();
    a: {
      var g = e;
      try {
        c.grow(g - m.byteLength + 65535 >>> 16);
        da(c.buffer);
        var h = 1;
        break a;
      } catch (k) {
        E("emscripten_realloc_buffer: Attempted to grow heap from " + m.byteLength + " bytes to " + g + " bytes, but got error: " + k);
      }
      h = void 0;
    }
    g = Ub();
    Ma("Heap resize call from " + b + " to " + e + " took " + (g - f) + " msecs. Success: " + !!h);
    if (h) {
      return !0;
    }
  }
  E("Failed to grow the heap from " + b + " bytes to " + e + " bytes, not enough memory!");
  return !1;
}, emscripten_set_canvas_element_size:function(a, b, d) {
  return Gc(a) ? Hc(a, b, d) : Ic(a, b, d);
}, emscripten_unwind_to_js_event_loop:function() {
  throw "unwind";
}, emscripten_webgl_create_context:function(a, b) {
  assert(b);
  b >>= 2;
  var d = t()[b + 6];
  b = {alpha:!!t()[b], depth:!!t()[b + 1], stencil:!!t()[b + 2], antialias:!!t()[b + 3], premultipliedAlpha:!!t()[b + 4], preserveDrawingBuffer:!!t()[b + 5], powerPreference:Qc[d], failIfMajorPerformanceCaveat:!!t()[b + 7], ua:t()[b + 8], Sa:t()[b + 9], U:t()[b + 10], ha:t()[b + 11], Va:t()[b + 12], Wa:t()[b + 13]};
  a = Gc(a);
  return !a || b.ha ? 0 : Mc(a, b);
}, environ_get:Uc, environ_sizes_get:Vc, exit:function(a) {
  Sb(a);
}, fd_close:Wc, fd_read:Xc, fd_seek:Yc, fd_write:Zc, getTempRet0:function() {
  return F;
}, gettimeofday:function(a) {
  var b = Date.now();
  H(a | 0, b / 1e3 | 0, 4);
  H(a + 4 | 0, b % 1e3 * 1e3 | 0, 4);
  return 0;
}, invoke_di:md, invoke_dii:nd, invoke_diii:od, invoke_fffffff:pd, invoke_fi:qd, invoke_fii:rd, invoke_fiii:sd, invoke_fijjjjifi:td, invoke_i:ud, invoke_if:vd, invoke_iffi:wd, invoke_ii:xd, invoke_iidd:yd, invoke_iidi:zd, invoke_iif:Ad, invoke_iiff:Bd, invoke_iii:Cd, invoke_iiifi:Dd, invoke_iiifii:Ed, invoke_iiii:Fd, invoke_iiiii:Gd, invoke_iiiiid:Hd, invoke_iiiiidfffiii:Id, invoke_iiiiifiiiiii:Jd, invoke_iiiiii:Kd, invoke_iiiiiii:Ld, invoke_iiiiiiii:Md, invoke_iiiiiiiii:Nd, invoke_iiiiiiiiii:Od, 
invoke_iiiiiiiiiii:Pd, invoke_iiiiiiiiiiii:Qd, invoke_iiiiiiiiiiiii:Rd, invoke_iiiiiiiiiiiiifi:Sd, invoke_iiiiiiiiiiiiiiii:Td, invoke_iiiiiiiiiiiiiiiiifi:Ud, invoke_iiiiiiiiiiiiiiiiiifi:Vd, invoke_iiiiiiiiiji:Wd, invoke_iiiiiiiiijii:Xd, invoke_iiiiiiiiijj:Yd, invoke_iiiiiiiijjji:Zd, invoke_iiiiiij:$d, invoke_iiiiiijji:ae, invoke_iiiiiijjjii:be, invoke_iiiiij:ce, invoke_iiiiiji:de, invoke_iiiiijiii:ee, invoke_iiiiijiiiii:fe, invoke_iiiiijji:ge, invoke_iiiij:he, invoke_iiiiji:ie, invoke_iiiijii:je, 
invoke_iiiijjii:ke, invoke_iiiijjj:le, invoke_iiij:me, invoke_iiiji:ne, invoke_iiijii:oe, invoke_iiijiii:pe, invoke_iiijiiiii:qe, invoke_iiijj:re, invoke_iiijjii:se, invoke_iij:te, invoke_iiji:ue, invoke_iijiiii:ve, invoke_iijj:we, invoke_iijjjf:xe, invoke_iijjjii:ye, invoke_ij:ze, invoke_iji:Ae, invoke_ijii:Be, invoke_ijiiii:Ce, invoke_ijjj:De, invoke_j:Ee, invoke_jfi:Fe, invoke_ji:Ge, invoke_jii:He, invoke_jiii:Ie, invoke_jiiii:Je, invoke_jiij:Ke, invoke_jiji:Le, invoke_jj:Me, invoke_jjj:Ne, invoke_v:Oe, 
invoke_vfiii:Pe, invoke_vi:Qe, invoke_vid:Re, invoke_vidi:Se, invoke_vif:Te, invoke_viffiii:Ue, invoke_vifi:Ve, invoke_vifii:We, invoke_vifiifiiii:Xe, invoke_vifiifiiiiiii:Ye, invoke_vifiii:Ze, invoke_vii:$e, invoke_viid:af, invoke_viidi:bf, invoke_viif:cf, invoke_viiff:df, invoke_viifi:ef, invoke_viifiifijjjii:ff, invoke_viifjjijiiii:gf, invoke_viifjjjijiiiii:hf, invoke_viii:jf, invoke_viiif:kf, invoke_viiiff:lf, invoke_viiifii:mf, invoke_viiifiifii:nf, invoke_viiifiiii:of, invoke_viiifiiiii:pf, 
invoke_viiifiiiiiifiiii:qf, invoke_viiii:rf, invoke_viiiiff:sf, invoke_viiiifiiifiii:tf, invoke_viiiii:uf, invoke_viiiiidiidii:vf, invoke_viiiiif:wf, invoke_viiiiiff:xf, invoke_viiiiifiifii:yf, invoke_viiiiifiiiifiii:zf, invoke_viiiiifiiiiii:Af, invoke_viiiiii:Bf, invoke_viiiiiid:Cf, invoke_viiiiiif:Df, invoke_viiiiiiff:Ef, invoke_viiiiiii:Ff, invoke_viiiiiiidiiii:Gf, invoke_viiiiiiifiiii:Hf, invoke_viiiiiiii:If, invoke_viiiiiiiii:Jf, invoke_viiiiiiiiii:Kf, invoke_viiiiiiiiiii:Lf, invoke_viiiiiiiiiiii:Mf, 
invoke_viiiiiiiiiiiii:Nf, invoke_viiiiiiiiiiiiii:Of, invoke_viiiiiiiiiiiiiifi:Pf, invoke_viiiiiiiiiiiiiii:Qf, invoke_viiiiiiiiiiiiiiii:Rf, invoke_viiiiiiiiiiiiiiiiiiiiiiiiii:Sf, invoke_viiiiiiiijjj:Tf, invoke_viiiiiiij:Uf, invoke_viiiiiiijiiii:Vf, invoke_viiiiiiijiiiiiiiiiiiiiiiii:Wf, invoke_viiiiiij:Xf, invoke_viiiiiijjjjjii:Yf, invoke_viiiiij:Zf, invoke_viiiiiji:$f, invoke_viiiiijiiiiii:ag, invoke_viiiiijjiiiii:bg, invoke_viiiij:cg, invoke_viiiiji:dg, invoke_viiiijii:eg, invoke_viiiijiiiiiiif:fg, 
invoke_viiiijiiiiiiii:gg, invoke_viiiijj:hg, invoke_viiiijjji:ig, invoke_viiij:jg, invoke_viiiji:kg, invoke_viiijii:lg, invoke_viiijiii:mg, invoke_viiijiiiiiiiii:ng, invoke_viiijiijjj:og, invoke_viiijj:pg, invoke_viiijjiiiiiii:qg, invoke_viiijjjfffi:rg, invoke_viiijjjfii:sg, invoke_viiijjjii:tg, invoke_viij:ug, invoke_viiji:vg, invoke_viijii:wg, invoke_viijiiii:xg, invoke_viijiiiiiiiii:yg, invoke_viijiiiiiiijjii:zg, invoke_viijiiiijii:Ag, invoke_viijj:Bg, invoke_viijjiii:Cg, invoke_viijjiiiiiiiii:Dg, 
invoke_viijjj:Eg, invoke_viijjjfiifiiii:Fg, invoke_viijjjiiiii:Gg, invoke_viijjjiiiiiii:Hg, invoke_viijjjj:Ig, invoke_viijjjjjjjjjjjjjif:Jg, invoke_viijjjjjjjjjjjjjii:Kg, invoke_vij:Lg, invoke_vijfjiiiii:Mg, invoke_viji:Ng, invoke_vijii:Og, invoke_vijiii:Pg, invoke_vijiiiiii:Qg, invoke_vijiiiiiiii:Rg, invoke_vijiji:Sg, invoke_vijjfffiii:Tg, invoke_vijjiiii:Ug, invoke_vijjj:Vg, invoke_vijjjiij:Wg, invoke_vijjjiiji:Xg, invoke_vijjjjiii:Yg, invoke_vijjjjjjjjjjjjjii:Zg, invoke_vj:$g, invoke_vjifiii:ah, 
invoke_vjiiiii:bh, invoke_vjiiiiii:ch, invoke_vjiiiiiii:dh, invoke_vjjfiii:eh, invoke_vjji:fh, invoke_vjjjjjjffiifiiiii:gh, invoke_vjjjjjjjjfffiifiiiii:hh, llvm_eh_typeid_for:function(a) {
  return a;
}, memory:c || u.wasmMemory, segfault:function() {
  v("segmentation fault");
}, setTempRet0:function(a) {
  F = a;
}, strftime:ed, strftime_l:function(a, b, d, e) {
  return ed(a, b, d, e);
}};
(function() {
  function a(g, h) {
    u.asm = g.exports;
    T.T.push(u.asm.emscripten_tls_init);
    lb = u.asm.__indirect_function_table;
    assert(lb, "table not found in wasm exports");
    sb.unshift(u.asm.__wasm_call_ctors);
    Za = h;
    B || (zb--, u.monitorRunDependencies && u.monitorRunDependencies(zb), assert(Cb["wasm-instantiate"]), delete Cb["wasm-instantiate"], 0 == zb && (null !== Ab && (clearInterval(Ab), Ab = null), Bb && (g = Bb, Bb = null, g())));
  }
  function b(g) {
    assert(u === f, "the Module object should not be replaced during async compilation - perhaps the order of HTML elements is wrong?");
    f = null;
    a(g.instance, g.module);
  }
  function d(g) {
    return Hb().then(function(h) {
      return WebAssembly.instantiate(h, e);
    }).then(function(h) {
      return h;
    }).then(g, function(h) {
      E("failed to asynchronously prepare wasm: " + h);
      K.startsWith("file://") && E("warning: Loading from a file URI (" + K + ") is not supported in most browsers. See https://emscripten.org/docs/getting_started/FAQ.html#how-do-i-run-a-local-webserver-for-testing-why-does-my-program-stall-in-downloading-or-preparing");
      v(h);
    });
  }
  var e = {env:ih, wasi_snapshot_preview1:ih};
  B || Db();
  var f = u;
  if (u.instantiateWasm) {
    try {
      return u.instantiateWasm(e, a);
    } catch (g) {
      return E("Module.instantiateWasm callback failed with error: " + g), !1;
    }
  }
  (function() {
    return Pa || "function" !== typeof WebAssembly.instantiateStreaming || Fb() || K.startsWith("file://") || "function" !== typeof fetch ? d(b) : fetch(K, {credentials:"same-origin"}).then(function(g) {
      return WebAssembly.instantiateStreaming(g, e).then(b, function(h) {
        E("wasm streaming compile failed: " + h);
        E("falling back to ArrayBuffer instantiation");
        return d(b);
      });
    });
  })().catch(ta);
  return {};
})();
u.___wasm_call_ctors = M("__wasm_call_ctors");
u._OrtInit = M("OrtInit");
u._OrtCreateSessionOptions = M("OrtCreateSessionOptions");
u._OrtAddSessionConfigEntry = M("OrtAddSessionConfigEntry");
u._OrtReleaseSessionOptions = M("OrtReleaseSessionOptions");
u._OrtCreateSession = M("OrtCreateSession");
u._OrtReleaseSession = M("OrtReleaseSession");
u._OrtGetInputCount = M("OrtGetInputCount");
u._OrtGetOutputCount = M("OrtGetOutputCount");
u._OrtGetInputName = M("OrtGetInputName");
u._OrtGetOutputName = M("OrtGetOutputName");
u._OrtFree = M("OrtFree");
u._OrtCreateTensor = M("OrtCreateTensor");
u._OrtGetTensorData = M("OrtGetTensorData");
u._OrtReleaseTensor = M("OrtReleaseTensor");
u._OrtCreateRunOptions = M("OrtCreateRunOptions");
u._OrtAddRunConfigEntry = M("OrtAddRunConfigEntry");
u._OrtReleaseRunOptions = M("OrtReleaseRunOptions");
u._OrtRun = M("OrtRun");
u._OrtEndProfiling = M("OrtEndProfiling");
var Wb = u.___errno_location = M("__errno_location"), Nb = u._pthread_self = M("pthread_self"), ib = u._malloc = M("malloc"), Zb = u._free = M("free");
u._emscripten_tls_init = M("emscripten_tls_init");
var jh = u.___funcs_on_exit = M("__funcs_on_exit"), Ob = u._emscripten_main_thread_process_queued_calls = M("emscripten_main_thread_process_queued_calls");
u.___dl_seterr = M("__dl_seterr");
var jd = u.__emscripten_thread_init = M("_emscripten_thread_init");
u._emscripten_current_thread_process_queued_calls = M("emscripten_current_thread_process_queued_calls");
u._emscripten_main_browser_thread_id = M("emscripten_main_browser_thread_id");
u._emscripten_sync_run_in_main_thread_2 = M("emscripten_sync_run_in_main_thread_2");
var ld = u._emscripten_sync_run_in_main_thread_4 = M("emscripten_sync_run_in_main_thread_4"), Bc = u._emscripten_run_in_main_runtime_thread_js = M("emscripten_run_in_main_runtime_thread_js"), Ec = u._emscripten_dispatch_to_thread_ = M("emscripten_dispatch_to_thread_"), Ya = u._emscripten_stack_get_base = function() {
  return (Ya = u._emscripten_stack_get_base = u.asm.emscripten_stack_get_base).apply(null, arguments);
}, nb = u._emscripten_stack_get_end = function() {
  return (nb = u._emscripten_stack_get_end = u.asm.emscripten_stack_get_end).apply(null, arguments);
}, Lb = u.__emscripten_thread_free_data = M("_emscripten_thread_free_data");
u.__emscripten_thread_exit = M("_emscripten_thread_exit");
var rc = u._memalign = M("memalign"), Xa = u._sbrk = M("sbrk");
u._emscripten_get_sbrk_ptr = M("emscripten_get_sbrk_ptr");
var X = u._setThrew = M("setThrew"), kh = u._emscripten_stack_init = function() {
  return (kh = u._emscripten_stack_init = u.asm.emscripten_stack_init).apply(null, arguments);
}, Qb = u._emscripten_stack_set_limits = function() {
  return (Qb = u._emscripten_stack_set_limits = u.asm.emscripten_stack_set_limits).apply(null, arguments);
};
u._emscripten_stack_get_free = function() {
  return (u._emscripten_stack_get_free = u.asm.emscripten_stack_get_free).apply(null, arguments);
};
var O = u.stackSave = M("stackSave"), S = u.stackRestore = M("stackRestore"), Ac = u.stackAlloc = M("stackAlloc");
u.___cxa_demangle = M("__cxa_demangle");
var hd = u.___cxa_can_catch = M("__cxa_can_catch"), $b = u.___cxa_is_pointer_type = M("__cxa_is_pointer_type"), lh = u.dynCall_ji = M("dynCall_ji"), mh = u.dynCall_viijiiiiiiiii = M("dynCall_viijiiiiiiiii"), nh = u.dynCall_viiij = M("dynCall_viiij"), oh = u.dynCall_jii = M("dynCall_jii"), ph = u.dynCall_jiji = M("dynCall_jiji"), qh = u.dynCall_iiiiiij = M("dynCall_iiiiiij"), rh = u.dynCall_iij = M("dynCall_iij"), sh = u.dynCall_vij = M("dynCall_vij"), th = u.dynCall_viiijii = M("dynCall_viiijii"), 
uh = u.dynCall_jj = M("dynCall_jj"), vh = u.dynCall_viiijiii = M("dynCall_viiijiii"), wh = u.dynCall_ij = M("dynCall_ij"), xh = u.dynCall_viijj = M("dynCall_viijj"), yh = u.dynCall_iiiiijiii = M("dynCall_iiiiijiii"), zh = u.dynCall_viij = M("dynCall_viij"), Ah = u.dynCall_iiiiijiiiii = M("dynCall_iiiiijiiiii"), Bh = u.dynCall_iiiiiijji = M("dynCall_iiiiiijji"), Ch = u.dynCall_vijiii = M("dynCall_vijiii"), Dh = u.dynCall_jjj = M("dynCall_jjj"), Eh = u.dynCall_viiji = M("dynCall_viiji"), Fh = u.dynCall_viiiji = 
M("dynCall_viiiji"), Gh = u.dynCall_viijjj = M("dynCall_viijjj"), Hh = u.dynCall_viifiifijjjii = M("dynCall_viifiifijjjii"), Ih = u.dynCall_jiii = M("dynCall_jiii"), Jh = u.dynCall_vijiji = M("dynCall_vijiji"), Kh = u.dynCall_jiij = M("dynCall_jiij"), Lh = u.dynCall_vjifiii = M("dynCall_vjifiii"), Mh = u.dynCall_iiji = M("dynCall_iiji"), Nh = u.dynCall_ijjj = M("dynCall_ijjj"), Oh = u.dynCall_viijjiiiiiiiii = M("dynCall_viijjiiiiiiiii"), Ph = u.dynCall_vijjjiiji = M("dynCall_vijjjiiji"), Qh = u.dynCall_viijiiiiiiijjii = 
M("dynCall_viijiiiiiiijjii"), Rh = u.dynCall_vjjfiii = M("dynCall_vjjfiii"), Sh = u.dynCall_iiijj = M("dynCall_iiijj"), Th = u.dynCall_vijjjjiii = M("dynCall_vijjjjiii"), Uh = u.dynCall_viiiijiiiiiiif = M("dynCall_viiiijiiiiiiif"), Vh = u.dynCall_viijjjfiifiiii = M("dynCall_viijjjfiifiiii"), Wh = u.dynCall_viiiiiiiijjj = M("dynCall_viiiiiiiijjj"), Xh = u.dynCall_viiiiiijjjjjii = M("dynCall_viiiiiijjjjjii"), Yh = u.dynCall_viiiijii = M("dynCall_viiiijii"), Zh = u.dynCall_viiiiij = M("dynCall_viiiiij"), 
$h = u.dynCall_iji = M("dynCall_iji"), ai = u.dynCall_vijjjjjjjjjjjjjii = M("dynCall_vijjjjjjjjjjjjjii"), bi = u.dynCall_viiijjiiiiiii = M("dynCall_viiijjiiiiiii"), ci = u.dynCall_viijiiiijii = M("dynCall_viijiiiijii"), di = u.dynCall_viifjjjijiiiii = M("dynCall_viifjjjijiiiii"), ei = u.dynCall_viifjjijiiii = M("dynCall_viifjjijiiii"), fi = u.dynCall_iiijiiiii = M("dynCall_iiijiiiii"), gi = u.dynCall_vj = M("dynCall_vj"), hi = u.dynCall_iiiiiji = M("dynCall_iiiiiji"), ii = u.dynCall_vjiiiii = M("dynCall_vjiiiii"), 
ji = u.dynCall_vjiiiiii = M("dynCall_vjiiiiii"), ki = u.dynCall_vijiiiiii = M("dynCall_vijiiiiii"), li = u.dynCall_vjiiiiiii = M("dynCall_vjiiiiiii"), mi = u.dynCall_viijjjjjjjjjjjjjif = M("dynCall_viijjjjjjjjjjjjjif"), ni = u.dynCall_viiiijj = M("dynCall_viiiijj"), oi = u.dynCall_viiiiiji = M("dynCall_viiiiiji"), pi = u.dynCall_j = M("dynCall_j"), qi = u.dynCall_viiijjjii = M("dynCall_viiijjjii"), ri = u.dynCall_iijj = M("dynCall_iijj"), si = u.dynCall_iiiij = M("dynCall_iiiij"), ti = u.dynCall_viiijjjfffi = 
M("dynCall_viiijjjfffi"), ui = u.dynCall_viiijiijjj = M("dynCall_viiijiijjj"), vi = u.dynCall_viijjjj = M("dynCall_viijjjj"), wi = u.dynCall_vjjjjjjffiifiiiii = M("dynCall_vjjjjjjffiifiiiii"), xi = u.dynCall_vjjjjjjjjfffiifiiiii = M("dynCall_vjjjjjjjjfffiifiiiii"), yi = u.dynCall_jfi = M("dynCall_jfi"), zi = u.dynCall_fijjjjifi = M("dynCall_fijjjjifi"), Ai = u.dynCall_vijjfffiii = M("dynCall_vijjfffiii"), Bi = u.dynCall_vijiiiiiiii = M("dynCall_vijiiiiiiii"), Ci = u.dynCall_viiijj = M("dynCall_viiijj"), 
Di = u.dynCall_viiiiijiiiiii = M("dynCall_viiiiijiiiiii"), Ei = u.dynCall_viiiiijjiiiii = M("dynCall_viiiiijjiiiii"), Fi = u.dynCall_viiiiji = M("dynCall_viiiiji"), Gi = u.dynCall_viijjiii = M("dynCall_viijjiii"), Hi = u.dynCall_vijii = M("dynCall_vijii"), Ii = u.dynCall_iiiiji = M("dynCall_iiiiji"), Ji = u.dynCall_viijjjjjjjjjjjjjii = M("dynCall_viijjjjjjjjjjjjjii"), Ki = u.dynCall_viiiijiiiiiiii = M("dynCall_viiiijiiiiiiii"), Li = u.dynCall_iijjjf = M("dynCall_iijjjf"), Mi = u.dynCall_viiiijjji = 
M("dynCall_viiiijjji");
u.dynCall_jjjjjj = M("dynCall_jjjjjj");
u.dynCall_jjjjjjj = M("dynCall_jjjjjjj");
var Ni = u.dynCall_vijjjiij = M("dynCall_vijjjiij"), Oi = u.dynCall_vijjj = M("dynCall_vijjj"), Pi = u.dynCall_viiiiiij = M("dynCall_viiiiiij"), Qi = u.dynCall_viiiiiiij = M("dynCall_viiiiiiij"), Ri = u.dynCall_viiijiiiiiiiii = M("dynCall_viiijiiiiiiiii"), Si = u.dynCall_iiiijjj = M("dynCall_iiiijjj"), Ti = u.dynCall_viijiiii = M("dynCall_viijiiii"), Ui = u.dynCall_iiijjii = M("dynCall_iiijjii");
u.dynCall_iijjii = M("dynCall_iijjii");
var Vi = u.dynCall_vijjiiii = M("dynCall_vijjiiii"), Wi = u.dynCall_viijjjiiiiiii = M("dynCall_viijjjiiiiiii"), Xi = u.dynCall_viijjjiiiii = M("dynCall_viijjjiiiii"), Yi = u.dynCall_viiijjjfii = M("dynCall_viiijjjfii"), Zi = u.dynCall_vijfjiiiii = M("dynCall_vijfjiiiii"), $i = u.dynCall_iiiiiiiiijj = M("dynCall_iiiiiiiiijj"), aj = u.dynCall_viiiiiiijiiiiiiiiiiiiiiiii = M("dynCall_viiiiiiijiiiiiiiiiiiiiiiii"), bj = u.dynCall_ijiiii = M("dynCall_ijiiii"), cj = u.dynCall_iiij = M("dynCall_iiij"), dj = 
u.dynCall_iiiji = M("dynCall_iiiji"), ej = u.dynCall_iiijii = M("dynCall_iiijii"), fj = u.dynCall_iiiiiiiiiji = M("dynCall_iiiiiiiiiji"), gj = u.dynCall_iiiiijji = M("dynCall_iiiiijji"), hj = u.dynCall_iiiijjii = M("dynCall_iiiijjii"), ij = u.dynCall_iiiijii = M("dynCall_iiiijii"), jj = u.dynCall_iiijiii = M("dynCall_iiijiii"), kj = u.dynCall_iiiiiiiiijii = M("dynCall_iiiiiiiiijii"), lj = u.dynCall_iiiiiijjjii = M("dynCall_iiiiiijjjii"), mj = u.dynCall_iiiiiiiijjji = M("dynCall_iiiiiiiijjji"), nj = 
u.dynCall_iijiiii = M("dynCall_iijiiii"), oj = u.dynCall_viiiij = M("dynCall_viiiij"), pj = u.dynCall_iijjjii = M("dynCall_iijjjii"), qj = u.dynCall_jiiii = M("dynCall_jiiii"), rj = u.dynCall_viijii = M("dynCall_viijii"), sj = u.dynCall_viji = M("dynCall_viji"), tj = u.dynCall_vjji = M("dynCall_vjji"), uj = u.dynCall_ijii = M("dynCall_ijii"), vj = u.dynCall_viiiiiiijiiii = M("dynCall_viiiiiiijiiii"), wj = u.dynCall_iiiiij = M("dynCall_iiiiij");
u.dynCall_iiiiijj = M("dynCall_iiiiijj");
u.dynCall_iiiiiijj = M("dynCall_iiiiiijj");
var kd = u.__emscripten_main_thread_futex = 1916660, Mb = u.__emscripten_allow_main_runtime_queued_calls = 1595756;
function Cd(a, b, d) {
  var e = O();
  try {
    return N(a)(b, d);
  } catch (f) {
    S(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    X(1, 0);
  }
}
function Qe(a, b) {
  var d = O();
  try {
    N(a)(b);
  } catch (e) {
    S(d);
    if (e !== e + 0 && "longjmp" !== e) {
      throw e;
    }
    X(1, 0);
  }
}
function xd(a, b) {
  var d = O();
  try {
    return N(a)(b);
  } catch (e) {
    S(d);
    if (e !== e + 0 && "longjmp" !== e) {
      throw e;
    }
    X(1, 0);
  }
}
function $e(a, b, d) {
  var e = O();
  try {
    N(a)(b, d);
  } catch (f) {
    S(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    X(1, 0);
  }
}
function jf(a, b, d, e) {
  var f = O();
  try {
    N(a)(b, d, e);
  } catch (g) {
    S(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    X(1, 0);
  }
}
function Fd(a, b, d, e) {
  var f = O();
  try {
    return N(a)(b, d, e);
  } catch (g) {
    S(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    X(1, 0);
  }
}
function Ld(a, b, d, e, f, g, h) {
  var k = O();
  try {
    return N(a)(b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function Oe(a) {
  var b = O();
  try {
    N(a)();
  } catch (d) {
    S(b);
    if (d !== d + 0 && "longjmp" !== d) {
      throw d;
    }
    X(1, 0);
  }
}
function rf(a, b, d, e, f) {
  var g = O();
  try {
    N(a)(b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function Kd(a, b, d, e, f, g) {
  var h = O();
  try {
    return N(a)(b, d, e, f, g);
  } catch (k) {
    S(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    X(1, 0);
  }
}
function Gd(a, b, d, e, f) {
  var g = O();
  try {
    return N(a)(b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function ud(a) {
  var b = O();
  try {
    return N(a)();
  } catch (d) {
    S(b);
    if (d !== d + 0 && "longjmp" !== d) {
      throw d;
    }
    X(1, 0);
  }
}
function nd(a, b, d) {
  var e = O();
  try {
    return N(a)(b, d);
  } catch (f) {
    S(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    X(1, 0);
  }
}
function Ff(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    N(a)(b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function Md(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    return N(a)(b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function uf(a, b, d, e, f, g) {
  var h = O();
  try {
    N(a)(b, d, e, f, g);
  } catch (k) {
    S(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    X(1, 0);
  }
}
function Bf(a, b, d, e, f, g, h) {
  var k = O();
  try {
    N(a)(b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function If(a, b, d, e, f, g, h, k, l) {
  var n = O();
  try {
    N(a)(b, d, e, f, g, h, k, l);
  } catch (q) {
    S(n);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    X(1, 0);
  }
}
function qd(a, b) {
  var d = O();
  try {
    return N(a)(b);
  } catch (e) {
    S(d);
    if (e !== e + 0 && "longjmp" !== e) {
      throw e;
    }
    X(1, 0);
  }
}
function Te(a, b, d) {
  var e = O();
  try {
    N(a)(b, d);
  } catch (f) {
    S(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    X(1, 0);
  }
}
function cf(a, b, d, e) {
  var f = O();
  try {
    N(a)(b, d, e);
  } catch (g) {
    S(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    X(1, 0);
  }
}
function Nd(a, b, d, e, f, g, h, k, l) {
  var n = O();
  try {
    return N(a)(b, d, e, f, g, h, k, l);
  } catch (q) {
    S(n);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    X(1, 0);
  }
}
function Kf(a, b, d, e, f, g, h, k, l, n, q) {
  var r = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q);
  } catch (y) {
    S(r);
    if (y !== y + 0 && "longjmp" !== y) {
      throw y;
    }
    X(1, 0);
  }
}
function Od(a, b, d, e, f, g, h, k, l, n) {
  var q = O();
  try {
    return N(a)(b, d, e, f, g, h, k, l, n);
  } catch (r) {
    S(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    X(1, 0);
  }
}
function Pd(a, b, d, e, f, g, h, k, l, n, q) {
  var r = O();
  try {
    return N(a)(b, d, e, f, g, h, k, l, n, q);
  } catch (y) {
    S(r);
    if (y !== y + 0 && "longjmp" !== y) {
      throw y;
    }
    X(1, 0);
  }
}
function Rd(a, b, d, e, f, g, h, k, l, n, q, r, y) {
  var p = O();
  try {
    return N(a)(b, d, e, f, g, h, k, l, n, q, r, y);
  } catch (w) {
    S(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    X(1, 0);
  }
}
function Lf(a, b, d, e, f, g, h, k, l, n, q, r) {
  var y = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r);
  } catch (p) {
    S(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    X(1, 0);
  }
}
function Nf(a, b, d, e, f, g, h, k, l, n, q, r, y, p) {
  var w = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r, y, p);
  } catch (z) {
    S(w);
    if (z !== z + 0 && "longjmp" !== z) {
      throw z;
    }
    X(1, 0);
  }
}
function Jf(a, b, d, e, f, g, h, k, l, n) {
  var q = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n);
  } catch (r) {
    S(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    X(1, 0);
  }
}
function Of(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w) {
  var z = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r, y, p, w);
  } catch (A) {
    S(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    X(1, 0);
  }
}
function Mf(a, b, d, e, f, g, h, k, l, n, q, r, y) {
  var p = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r, y);
  } catch (w) {
    S(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    X(1, 0);
  }
}
function Xe(a, b, d, e, f, g, h, k, l, n) {
  var q = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n);
  } catch (r) {
    S(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    X(1, 0);
  }
}
function Ye(a, b, d, e, f, g, h, k, l, n, q, r, y) {
  var p = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r, y);
  } catch (w) {
    S(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    X(1, 0);
  }
}
function bf(a, b, d, e, f) {
  var g = O();
  try {
    N(a)(b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function wd(a, b, d, e) {
  var f = O();
  try {
    return N(a)(b, d, e);
  } catch (g) {
    S(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    X(1, 0);
  }
}
function sd(a, b, d, e) {
  var f = O();
  try {
    return N(a)(b, d, e);
  } catch (g) {
    S(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    X(1, 0);
  }
}
function Id(a, b, d, e, f, g, h, k, l, n, q, r) {
  var y = O();
  try {
    return N(a)(b, d, e, f, g, h, k, l, n, q, r);
  } catch (p) {
    S(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    X(1, 0);
  }
}
function Qf(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z) {
  var A = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r, y, p, w, z);
  } catch (C) {
    S(A);
    if (C !== C + 0 && "longjmp" !== C) {
      throw C;
    }
    X(1, 0);
  }
}
function Rf(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A) {
  var C = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A);
  } catch (G) {
    S(C);
    if (G !== G + 0 && "longjmp" !== G) {
      throw G;
    }
    X(1, 0);
  }
}
function mf(a, b, d, e, f, g, h) {
  var k = O();
  try {
    N(a)(b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function Af(a, b, d, e, f, g, h, k, l, n, q, r, y) {
  var p = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r, y);
  } catch (w) {
    S(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    X(1, 0);
  }
}
function wf(a, b, d, e, f, g, h) {
  var k = O();
  try {
    N(a)(b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function lf(a, b, d, e, f, g) {
  var h = O();
  try {
    N(a)(b, d, e, f, g);
  } catch (k) {
    S(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    X(1, 0);
  }
}
function Ef(a, b, d, e, f, g, h, k, l) {
  var n = O();
  try {
    N(a)(b, d, e, f, g, h, k, l);
  } catch (q) {
    S(n);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    X(1, 0);
  }
}
function xf(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    N(a)(b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function Ud(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G) {
  var I = O();
  try {
    return N(a)(b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G);
  } catch (L) {
    S(I);
    if (L !== L + 0 && "longjmp" !== L) {
      throw L;
    }
    X(1, 0);
  }
}
function Ad(a, b, d) {
  var e = O();
  try {
    return N(a)(b, d);
  } catch (f) {
    S(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    X(1, 0);
  }
}
function vf(a, b, d, e, f, g, h, k, l, n, q, r) {
  var y = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r);
  } catch (p) {
    S(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    X(1, 0);
  }
}
function nf(a, b, d, e, f, g, h, k, l, n) {
  var q = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n);
  } catch (r) {
    S(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    X(1, 0);
  }
}
function Td(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z) {
  var A = O();
  try {
    return N(a)(b, d, e, f, g, h, k, l, n, q, r, y, p, w, z);
  } catch (C) {
    S(A);
    if (C !== C + 0 && "longjmp" !== C) {
      throw C;
    }
    X(1, 0);
  }
}
function Ed(a, b, d, e, f, g) {
  var h = O();
  try {
    return N(a)(b, d, e, f, g);
  } catch (k) {
    S(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    X(1, 0);
  }
}
function Jd(a, b, d, e, f, g, h, k, l, n, q, r) {
  var y = O();
  try {
    return N(a)(b, d, e, f, g, h, k, l, n, q, r);
  } catch (p) {
    S(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    X(1, 0);
  }
}
function Pf(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A) {
  var C = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A);
  } catch (G) {
    S(C);
    if (G !== G + 0 && "longjmp" !== G) {
      throw G;
    }
    X(1, 0);
  }
}
function yf(a, b, d, e, f, g, h, k, l, n, q, r) {
  var y = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r);
  } catch (p) {
    S(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    X(1, 0);
  }
}
function tf(a, b, d, e, f, g, h, k, l, n, q, r, y) {
  var p = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r, y);
  } catch (w) {
    S(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    X(1, 0);
  }
}
function qf(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z) {
  var A = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r, y, p, w, z);
  } catch (C) {
    S(A);
    if (C !== C + 0 && "longjmp" !== C) {
      throw C;
    }
    X(1, 0);
  }
}
function Pe(a, b, d, e, f) {
  var g = O();
  try {
    N(a)(b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function df(a, b, d, e, f) {
  var g = O();
  try {
    N(a)(b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function Vd(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G, I) {
  var L = O();
  try {
    return N(a)(b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G, I);
  } catch (P) {
    S(L);
    if (P !== P + 0 && "longjmp" !== P) {
      throw P;
    }
    X(1, 0);
  }
}
function zf(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w) {
  var z = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r, y, p, w);
  } catch (A) {
    S(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    X(1, 0);
  }
}
function od(a, b, d, e) {
  var f = O();
  try {
    return N(a)(b, d, e);
  } catch (g) {
    S(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    X(1, 0);
  }
}
function Ve(a, b, d, e) {
  var f = O();
  try {
    N(a)(b, d, e);
  } catch (g) {
    S(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    X(1, 0);
  }
}
function Df(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    N(a)(b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function Hf(a, b, d, e, f, g, h, k, l, n, q, r, y) {
  var p = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r, y);
  } catch (w) {
    S(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    X(1, 0);
  }
}
function md(a, b) {
  var d = O();
  try {
    return N(a)(b);
  } catch (e) {
    S(d);
    if (e !== e + 0 && "longjmp" !== e) {
      throw e;
    }
    X(1, 0);
  }
}
function Cf(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    N(a)(b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function Gf(a, b, d, e, f, g, h, k, l, n, q, r, y) {
  var p = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r, y);
  } catch (w) {
    S(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    X(1, 0);
  }
}
function rd(a, b, d) {
  var e = O();
  try {
    return N(a)(b, d);
  } catch (f) {
    S(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    X(1, 0);
  }
}
function Qd(a, b, d, e, f, g, h, k, l, n, q, r) {
  var y = O();
  try {
    return N(a)(b, d, e, f, g, h, k, l, n, q, r);
  } catch (p) {
    S(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    X(1, 0);
  }
}
function Ze(a, b, d, e, f, g) {
  var h = O();
  try {
    N(a)(b, d, e, f, g);
  } catch (k) {
    S(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    X(1, 0);
  }
}
function We(a, b, d, e, f) {
  var g = O();
  try {
    N(a)(b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function Re(a, b, d) {
  var e = O();
  try {
    N(a)(b, d);
  } catch (f) {
    S(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    X(1, 0);
  }
}
function Ue(a, b, d, e, f, g, h) {
  var k = O();
  try {
    N(a)(b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function yd(a, b, d, e) {
  var f = O();
  try {
    return N(a)(b, d, e);
  } catch (g) {
    S(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    X(1, 0);
  }
}
function af(a, b, d, e) {
  var f = O();
  try {
    N(a)(b, d, e);
  } catch (g) {
    S(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    X(1, 0);
  }
}
function Sd(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w) {
  var z = O();
  try {
    return N(a)(b, d, e, f, g, h, k, l, n, q, r, y, p, w);
  } catch (A) {
    S(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    X(1, 0);
  }
}
function sf(a, b, d, e, f, g, h) {
  var k = O();
  try {
    N(a)(b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function pd(a, b, d, e, f, g, h) {
  var k = O();
  try {
    return N(a)(b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function Bd(a, b, d, e) {
  var f = O();
  try {
    return N(a)(b, d, e);
  } catch (g) {
    S(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    X(1, 0);
  }
}
function kf(a, b, d, e, f) {
  var g = O();
  try {
    N(a)(b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function Sf(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G, I, L, P, U, V, R, Y, Z) {
  var aa = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G, I, L, P, U, V, R, Y, Z);
  } catch (Q) {
    S(aa);
    if (Q !== Q + 0 && "longjmp" !== Q) {
      throw Q;
    }
    X(1, 0);
  }
}
function zd(a, b, d, e) {
  var f = O();
  try {
    return N(a)(b, d, e);
  } catch (g) {
    S(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    X(1, 0);
  }
}
function vd(a, b) {
  var d = O();
  try {
    return N(a)(b);
  } catch (e) {
    S(d);
    if (e !== e + 0 && "longjmp" !== e) {
      throw e;
    }
    X(1, 0);
  }
}
function Dd(a, b, d, e, f) {
  var g = O();
  try {
    return N(a)(b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function Se(a, b, d, e) {
  var f = O();
  try {
    N(a)(b, d, e);
  } catch (g) {
    S(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    X(1, 0);
  }
}
function ef(a, b, d, e, f) {
  var g = O();
  try {
    N(a)(b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function pf(a, b, d, e, f, g, h, k, l, n) {
  var q = O();
  try {
    N(a)(b, d, e, f, g, h, k, l, n);
  } catch (r) {
    S(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    X(1, 0);
  }
}
function of(a, b, d, e, f, g, h, k, l) {
  var n = O();
  try {
    N(a)(b, d, e, f, g, h, k, l);
  } catch (q) {
    S(n);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    X(1, 0);
  }
}
function Hd(a, b, d, e, f, g) {
  var h = O();
  try {
    return N(a)(b, d, e, f, g);
  } catch (k) {
    S(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    X(1, 0);
  }
}
function Le(a, b, d, e, f) {
  var g = O();
  try {
    return ph(a, b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function $d(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    return qh(a, b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function He(a, b, d) {
  var e = O();
  try {
    return oh(a, b, d);
  } catch (f) {
    S(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    X(1, 0);
  }
}
function Ge(a, b) {
  var d = O();
  try {
    return lh(a, b);
  } catch (e) {
    S(d);
    if (e !== e + 0 && "longjmp" !== e) {
      throw e;
    }
    X(1, 0);
  }
}
function te(a, b, d, e) {
  var f = O();
  try {
    return rh(a, b, d, e);
  } catch (g) {
    S(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    X(1, 0);
  }
}
function Lg(a, b, d, e) {
  var f = O();
  try {
    sh(a, b, d, e);
  } catch (g) {
    S(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    X(1, 0);
  }
}
function Ne(a, b, d, e, f) {
  var g = O();
  try {
    return Dh(a, b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function mg(a, b, d, e, f, g, h, k, l) {
  var n = O();
  try {
    vh(a, b, d, e, f, g, h, k, l);
  } catch (q) {
    S(n);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    X(1, 0);
  }
}
function lg(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    th(a, b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function ze(a, b, d) {
  var e = O();
  try {
    return wh(a, b, d);
  } catch (f) {
    S(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    X(1, 0);
  }
}
function Bg(a, b, d, e, f, g, h) {
  var k = O();
  try {
    xh(a, b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function ug(a, b, d, e, f) {
  var g = O();
  try {
    zh(a, b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function fe(a, b, d, e, f, g, h, k, l, n, q, r) {
  var y = O();
  try {
    return Ah(a, b, d, e, f, g, h, k, l, n, q, r);
  } catch (p) {
    S(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    X(1, 0);
  }
}
function ee(a, b, d, e, f, g, h, k, l, n) {
  var q = O();
  try {
    return yh(a, b, d, e, f, g, h, k, l, n);
  } catch (r) {
    S(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    X(1, 0);
  }
}
function ae(a, b, d, e, f, g, h, k, l, n, q) {
  var r = O();
  try {
    return Bh(a, b, d, e, f, g, h, k, l, n, q);
  } catch (y) {
    S(r);
    if (y !== y + 0 && "longjmp" !== y) {
      throw y;
    }
    X(1, 0);
  }
}
function Ng(a, b, d, e, f) {
  var g = O();
  try {
    sj(a, b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function Pg(a, b, d, e, f, g, h) {
  var k = O();
  try {
    Ch(a, b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function vg(a, b, d, e, f, g) {
  var h = O();
  try {
    Eh(a, b, d, e, f, g);
  } catch (k) {
    S(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    X(1, 0);
  }
}
function kg(a, b, d, e, f, g, h) {
  var k = O();
  try {
    Fh(a, b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function Eg(a, b, d, e, f, g, h, k, l) {
  var n = O();
  try {
    Gh(a, b, d, e, f, g, h, k, l);
  } catch (q) {
    S(n);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    X(1, 0);
  }
}
function ff(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z) {
  var A = O();
  try {
    Hh(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z);
  } catch (C) {
    S(A);
    if (C !== C + 0 && "longjmp" !== C) {
      throw C;
    }
    X(1, 0);
  }
}
function Ie(a, b, d, e) {
  var f = O();
  try {
    return Ih(a, b, d, e);
  } catch (g) {
    S(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    X(1, 0);
  }
}
function Sg(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    Jh(a, b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function Ke(a, b, d, e, f) {
  var g = O();
  try {
    return Kh(a, b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function ah(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    Lh(a, b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function ue(a, b, d, e, f) {
  var g = O();
  try {
    return Mh(a, b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function De(a, b, d, e, f, g, h) {
  var k = O();
  try {
    return Nh(a, b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function Dg(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z) {
  var A = O();
  try {
    Oh(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z);
  } catch (C) {
    S(A);
    if (C !== C + 0 && "longjmp" !== C) {
      throw C;
    }
    X(1, 0);
  }
}
function Xg(a, b, d, e, f, g, h, k, l, n, q, r, y) {
  var p = O();
  try {
    Ph(a, b, d, e, f, g, h, k, l, n, q, r, y);
  } catch (w) {
    S(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    X(1, 0);
  }
}
function zg(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C) {
  var G = O();
  try {
    Qh(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C);
  } catch (I) {
    S(G);
    if (I !== I + 0 && "longjmp" !== I) {
      throw I;
    }
    X(1, 0);
  }
}
function eh(a, b, d, e, f, g, h, k, l) {
  var n = O();
  try {
    Rh(a, b, d, e, f, g, h, k, l);
  } catch (q) {
    S(n);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    X(1, 0);
  }
}
function re(a, b, d, e, f, g, h) {
  var k = O();
  try {
    return Sh(a, b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function Yg(a, b, d, e, f, g, h, k, l, n, q, r, y) {
  var p = O();
  try {
    Th(a, b, d, e, f, g, h, k, l, n, q, r, y);
  } catch (w) {
    S(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    X(1, 0);
  }
}
function fg(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w) {
  var z = O();
  try {
    Uh(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w);
  } catch (A) {
    S(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    X(1, 0);
  }
}
function Fg(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A) {
  var C = O();
  try {
    Vh(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A);
  } catch (G) {
    S(C);
    if (G !== G + 0 && "longjmp" !== G) {
      throw G;
    }
    X(1, 0);
  }
}
function Tf(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w) {
  var z = O();
  try {
    Wh(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w);
  } catch (A) {
    S(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    X(1, 0);
  }
}
function Yf(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G) {
  var I = O();
  try {
    Xh(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G);
  } catch (L) {
    S(I);
    if (L !== L + 0 && "longjmp" !== L) {
      throw L;
    }
    X(1, 0);
  }
}
function eg(a, b, d, e, f, g, h, k, l) {
  var n = O();
  try {
    Yh(a, b, d, e, f, g, h, k, l);
  } catch (q) {
    S(n);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    X(1, 0);
  }
}
function Zf(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    Zh(a, b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function Ae(a, b, d, e) {
  var f = O();
  try {
    return $h(a, b, d, e);
  } catch (g) {
    S(f);
    if (g !== g + 0 && "longjmp" !== g) {
      throw g;
    }
    X(1, 0);
  }
}
function Zg(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G, I, L, P, U, V, R, Y, Z, aa, Q, ca) {
  var Ga = O();
  try {
    ai(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G, I, L, P, U, V, R, Y, Z, aa, Q, ca);
  } catch (ja) {
    S(Ga);
    if (ja !== ja + 0 && "longjmp" !== ja) {
      throw ja;
    }
    X(1, 0);
  }
}
function qg(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w) {
  var z = O();
  try {
    bi(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w);
  } catch (A) {
    S(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    X(1, 0);
  }
}
function Ag(a, b, d, e, f, g, h, k, l, n, q, r, y) {
  var p = O();
  try {
    ci(a, b, d, e, f, g, h, k, l, n, q, r, y);
  } catch (w) {
    S(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    X(1, 0);
  }
}
function hf(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C) {
  var G = O();
  try {
    di(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C);
  } catch (I) {
    S(G);
    if (I !== I + 0 && "longjmp" !== I) {
      throw I;
    }
    X(1, 0);
  }
}
function gf(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w) {
  var z = O();
  try {
    ei(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w);
  } catch (A) {
    S(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    X(1, 0);
  }
}
function qe(a, b, d, e, f, g, h, k, l, n) {
  var q = O();
  try {
    return fi(a, b, d, e, f, g, h, k, l, n);
  } catch (r) {
    S(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    X(1, 0);
  }
}
function $g(a, b, d) {
  var e = O();
  try {
    gi(a, b, d);
  } catch (f) {
    S(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    X(1, 0);
  }
}
function de(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    return hi(a, b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function bh(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    ii(a, b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function ch(a, b, d, e, f, g, h, k, l) {
  var n = O();
  try {
    ji(a, b, d, e, f, g, h, k, l);
  } catch (q) {
    S(n);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    X(1, 0);
  }
}
function Qg(a, b, d, e, f, g, h, k, l, n) {
  var q = O();
  try {
    ki(a, b, d, e, f, g, h, k, l, n);
  } catch (r) {
    S(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    X(1, 0);
  }
}
function dh(a, b, d, e, f, g, h, k, l, n) {
  var q = O();
  try {
    li(a, b, d, e, f, g, h, k, l, n);
  } catch (r) {
    S(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    X(1, 0);
  }
}
function Jg(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G, I, L, P, U, V, R, Y, Z, aa, Q, ca, Ga) {
  var ja = O();
  try {
    mi(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G, I, L, P, U, V, R, Y, Z, aa, Q, ca, Ga);
  } catch (fa) {
    S(ja);
    if (fa !== fa + 0 && "longjmp" !== fa) {
      throw fa;
    }
    X(1, 0);
  }
}
function jg(a, b, d, e, f, g) {
  var h = O();
  try {
    nh(a, b, d, e, f, g);
  } catch (k) {
    S(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    X(1, 0);
  }
}
function hg(a, b, d, e, f, g, h, k, l) {
  var n = O();
  try {
    ni(a, b, d, e, f, g, h, k, l);
  } catch (q) {
    S(n);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    X(1, 0);
  }
}
function $f(a, b, d, e, f, g, h, k, l) {
  var n = O();
  try {
    oi(a, b, d, e, f, g, h, k, l);
  } catch (q) {
    S(n);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    X(1, 0);
  }
}
function tg(a, b, d, e, f, g, h, k, l, n, q, r) {
  var y = O();
  try {
    qi(a, b, d, e, f, g, h, k, l, n, q, r);
  } catch (p) {
    S(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    X(1, 0);
  }
}
function we(a, b, d, e, f, g) {
  var h = O();
  try {
    return ri(a, b, d, e, f, g);
  } catch (k) {
    S(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    X(1, 0);
  }
}
function Ee(a) {
  var b = O();
  try {
    return pi(a);
  } catch (d) {
    S(b);
    if (d !== d + 0 && "longjmp" !== d) {
      throw d;
    }
    X(1, 0);
  }
}
function he(a, b, d, e, f, g) {
  var h = O();
  try {
    return si(a, b, d, e, f, g);
  } catch (k) {
    S(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    X(1, 0);
  }
}
function rg(a, b, d, e, f, g, h, k, l, n, q, r, y, p) {
  var w = O();
  try {
    ti(a, b, d, e, f, g, h, k, l, n, q, r, y, p);
  } catch (z) {
    S(w);
    if (z !== z + 0 && "longjmp" !== z) {
      throw z;
    }
    X(1, 0);
  }
}
function og(a, b, d, e, f, g, h, k, l, n, q, r, y, p) {
  var w = O();
  try {
    ui(a, b, d, e, f, g, h, k, l, n, q, r, y, p);
  } catch (z) {
    S(w);
    if (z !== z + 0 && "longjmp" !== z) {
      throw z;
    }
    X(1, 0);
  }
}
function Ig(a, b, d, e, f, g, h, k, l, n, q) {
  var r = O();
  try {
    vi(a, b, d, e, f, g, h, k, l, n, q);
  } catch (y) {
    S(r);
    if (y !== y + 0 && "longjmp" !== y) {
      throw y;
    }
    X(1, 0);
  }
}
function gh(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G, I, L, P, U) {
  var V = O();
  try {
    wi(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G, I, L, P, U);
  } catch (R) {
    S(V);
    if (R !== R + 0 && "longjmp" !== R) {
      throw R;
    }
    X(1, 0);
  }
}
function hh(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G, I, L, P, U, V, R, Y, Z, aa) {
  var Q = O();
  try {
    xi(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G, I, L, P, U, V, R, Y, Z, aa);
  } catch (ca) {
    S(Q);
    if (ca !== ca + 0 && "longjmp" !== ca) {
      throw ca;
    }
    X(1, 0);
  }
}
function td(a, b, d, e, f, g, h, k, l, n, q, r, y) {
  var p = O();
  try {
    return zi(a, b, d, e, f, g, h, k, l, n, q, r, y);
  } catch (w) {
    S(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    X(1, 0);
  }
}
function Tg(a, b, d, e, f, g, h, k, l, n, q, r) {
  var y = O();
  try {
    Ai(a, b, d, e, f, g, h, k, l, n, q, r);
  } catch (p) {
    S(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    X(1, 0);
  }
}
function Rg(a, b, d, e, f, g, h, k, l, n, q, r) {
  var y = O();
  try {
    Bi(a, b, d, e, f, g, h, k, l, n, q, r);
  } catch (p) {
    S(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    X(1, 0);
  }
}
function Fe(a, b, d) {
  var e = O();
  try {
    return yi(a, b, d);
  } catch (f) {
    S(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    X(1, 0);
  }
}
function pg(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    Ci(a, b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function ag(a, b, d, e, f, g, h, k, l, n, q, r, y, p) {
  var w = O();
  try {
    Di(a, b, d, e, f, g, h, k, l, n, q, r, y, p);
  } catch (z) {
    S(w);
    if (z !== z + 0 && "longjmp" !== z) {
      throw z;
    }
    X(1, 0);
  }
}
function bg(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w) {
  var z = O();
  try {
    Ei(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w);
  } catch (A) {
    S(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    X(1, 0);
  }
}
function dg(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    Fi(a, b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function Cg(a, b, d, e, f, g, h, k, l, n) {
  var q = O();
  try {
    Gi(a, b, d, e, f, g, h, k, l, n);
  } catch (r) {
    S(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    X(1, 0);
  }
}
function Og(a, b, d, e, f, g) {
  var h = O();
  try {
    Hi(a, b, d, e, f, g);
  } catch (k) {
    S(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    X(1, 0);
  }
}
function ie(a, b, d, e, f, g, h) {
  var k = O();
  try {
    return Ii(a, b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function Kg(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G, I, L, P, U, V, R, Y, Z, aa, Q, ca, Ga) {
  var ja = O();
  try {
    Ji(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G, I, L, P, U, V, R, Y, Z, aa, Q, ca, Ga);
  } catch (fa) {
    S(ja);
    if (fa !== fa + 0 && "longjmp" !== fa) {
      throw fa;
    }
    X(1, 0);
  }
}
function gg(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w) {
  var z = O();
  try {
    Ki(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w);
  } catch (A) {
    S(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    X(1, 0);
  }
}
function xe(a, b, d, e, f, g, h, k, l) {
  var n = O();
  try {
    return Li(a, b, d, e, f, g, h, k, l);
  } catch (q) {
    S(n);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    X(1, 0);
  }
}
function ig(a, b, d, e, f, g, h, k, l, n, q, r) {
  var y = O();
  try {
    Mi(a, b, d, e, f, g, h, k, l, n, q, r);
  } catch (p) {
    S(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    X(1, 0);
  }
}
function Wg(a, b, d, e, f, g, h, k, l, n, q, r) {
  var y = O();
  try {
    Ni(a, b, d, e, f, g, h, k, l, n, q, r);
  } catch (p) {
    S(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    X(1, 0);
  }
}
function Vg(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    Oi(a, b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function Xf(a, b, d, e, f, g, h, k, l) {
  var n = O();
  try {
    Pi(a, b, d, e, f, g, h, k, l);
  } catch (q) {
    S(n);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    X(1, 0);
  }
}
function Uf(a, b, d, e, f, g, h, k, l, n) {
  var q = O();
  try {
    Qi(a, b, d, e, f, g, h, k, l, n);
  } catch (r) {
    S(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    X(1, 0);
  }
}
function ng(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w) {
  var z = O();
  try {
    Ri(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w);
  } catch (A) {
    S(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    X(1, 0);
  }
}
function le(a, b, d, e, f, g, h, k, l, n) {
  var q = O();
  try {
    return Si(a, b, d, e, f, g, h, k, l, n);
  } catch (r) {
    S(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    X(1, 0);
  }
}
function xg(a, b, d, e, f, g, h, k, l) {
  var n = O();
  try {
    Ti(a, b, d, e, f, g, h, k, l);
  } catch (q) {
    S(n);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    X(1, 0);
  }
}
function se(a, b, d, e, f, g, h, k, l) {
  var n = O();
  try {
    return Ui(a, b, d, e, f, g, h, k, l);
  } catch (q) {
    S(n);
    if (q !== q + 0 && "longjmp" !== q) {
      throw q;
    }
    X(1, 0);
  }
}
function Ug(a, b, d, e, f, g, h, k, l, n) {
  var q = O();
  try {
    Vi(a, b, d, e, f, g, h, k, l, n);
  } catch (r) {
    S(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    X(1, 0);
  }
}
function Hg(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z) {
  var A = O();
  try {
    Wi(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z);
  } catch (C) {
    S(A);
    if (C !== C + 0 && "longjmp" !== C) {
      throw C;
    }
    X(1, 0);
  }
}
function Gg(a, b, d, e, f, g, h, k, l, n, q, r, y, p) {
  var w = O();
  try {
    Xi(a, b, d, e, f, g, h, k, l, n, q, r, y, p);
  } catch (z) {
    S(w);
    if (z !== z + 0 && "longjmp" !== z) {
      throw z;
    }
    X(1, 0);
  }
}
function sg(a, b, d, e, f, g, h, k, l, n, q, r, y) {
  var p = O();
  try {
    Yi(a, b, d, e, f, g, h, k, l, n, q, r, y);
  } catch (w) {
    S(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    X(1, 0);
  }
}
function Mg(a, b, d, e, f, g, h, k, l, n, q, r) {
  var y = O();
  try {
    Zi(a, b, d, e, f, g, h, k, l, n, q, r);
  } catch (p) {
    S(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    X(1, 0);
  }
}
function Yd(a, b, d, e, f, g, h, k, l, n, q, r, y) {
  var p = O();
  try {
    return $i(a, b, d, e, f, g, h, k, l, n, q, r, y);
  } catch (w) {
    S(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    X(1, 0);
  }
}
function Wf(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G, I, L, P, U, V, R, Y, Z) {
  var aa = O();
  try {
    aj(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w, z, A, C, G, I, L, P, U, V, R, Y, Z);
  } catch (Q) {
    S(aa);
    if (Q !== Q + 0 && "longjmp" !== Q) {
      throw Q;
    }
    X(1, 0);
  }
}
function Ce(a, b, d, e, f, g, h) {
  var k = O();
  try {
    return bj(a, b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function yg(a, b, d, e, f, g, h, k, l, n, q, r, y, p) {
  var w = O();
  try {
    mh(a, b, d, e, f, g, h, k, l, n, q, r, y, p);
  } catch (z) {
    S(w);
    if (z !== z + 0 && "longjmp" !== z) {
      throw z;
    }
    X(1, 0);
  }
}
function me(a, b, d, e, f) {
  var g = O();
  try {
    return cj(a, b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function ne(a, b, d, e, f, g) {
  var h = O();
  try {
    return dj(a, b, d, e, f, g);
  } catch (k) {
    S(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    X(1, 0);
  }
}
function oe(a, b, d, e, f, g, h) {
  var k = O();
  try {
    return ej(a, b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function Wd(a, b, d, e, f, g, h, k, l, n, q, r) {
  var y = O();
  try {
    return fj(a, b, d, e, f, g, h, k, l, n, q, r);
  } catch (p) {
    S(y);
    if (p !== p + 0 && "longjmp" !== p) {
      throw p;
    }
    X(1, 0);
  }
}
function ge(a, b, d, e, f, g, h, k, l, n) {
  var q = O();
  try {
    return gj(a, b, d, e, f, g, h, k, l, n);
  } catch (r) {
    S(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    X(1, 0);
  }
}
function ke(a, b, d, e, f, g, h, k, l, n) {
  var q = O();
  try {
    return hj(a, b, d, e, f, g, h, k, l, n);
  } catch (r) {
    S(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    X(1, 0);
  }
}
function je(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    return ij(a, b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function pe(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    return jj(a, b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function Xd(a, b, d, e, f, g, h, k, l, n, q, r, y) {
  var p = O();
  try {
    return kj(a, b, d, e, f, g, h, k, l, n, q, r, y);
  } catch (w) {
    S(p);
    if (w !== w + 0 && "longjmp" !== w) {
      throw w;
    }
    X(1, 0);
  }
}
function be(a, b, d, e, f, g, h, k, l, n, q, r, y, p) {
  var w = O();
  try {
    return lj(a, b, d, e, f, g, h, k, l, n, q, r, y, p);
  } catch (z) {
    S(w);
    if (z !== z + 0 && "longjmp" !== z) {
      throw z;
    }
    X(1, 0);
  }
}
function Zd(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w) {
  var z = O();
  try {
    return mj(a, b, d, e, f, g, h, k, l, n, q, r, y, p, w);
  } catch (A) {
    S(z);
    if (A !== A + 0 && "longjmp" !== A) {
      throw A;
    }
    X(1, 0);
  }
}
function ve(a, b, d, e, f, g, h, k) {
  var l = O();
  try {
    return nj(a, b, d, e, f, g, h, k);
  } catch (n) {
    S(l);
    if (n !== n + 0 && "longjmp" !== n) {
      throw n;
    }
    X(1, 0);
  }
}
function cg(a, b, d, e, f, g, h) {
  var k = O();
  try {
    oj(a, b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function ye(a, b, d, e, f, g, h, k, l, n) {
  var q = O();
  try {
    return pj(a, b, d, e, f, g, h, k, l, n);
  } catch (r) {
    S(q);
    if (r !== r + 0 && "longjmp" !== r) {
      throw r;
    }
    X(1, 0);
  }
}
function Me(a, b, d) {
  var e = O();
  try {
    return uh(a, b, d);
  } catch (f) {
    S(e);
    if (f !== f + 0 && "longjmp" !== f) {
      throw f;
    }
    X(1, 0);
  }
}
function Je(a, b, d, e, f) {
  var g = O();
  try {
    return qj(a, b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function wg(a, b, d, e, f, g, h) {
  var k = O();
  try {
    rj(a, b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
function fh(a, b, d, e, f, g) {
  var h = O();
  try {
    tj(a, b, d, e, f, g);
  } catch (k) {
    S(h);
    if (k !== k + 0 && "longjmp" !== k) {
      throw k;
    }
    X(1, 0);
  }
}
function Be(a, b, d, e, f) {
  var g = O();
  try {
    return uj(a, b, d, e, f);
  } catch (h) {
    S(g);
    if (h !== h + 0 && "longjmp" !== h) {
      throw h;
    }
    X(1, 0);
  }
}
function Vf(a, b, d, e, f, g, h, k, l, n, q, r, y, p) {
  var w = O();
  try {
    vj(a, b, d, e, f, g, h, k, l, n, q, r, y, p);
  } catch (z) {
    S(w);
    if (z !== z + 0 && "longjmp" !== z) {
      throw z;
    }
    X(1, 0);
  }
}
function ce(a, b, d, e, f, g, h) {
  var k = O();
  try {
    return wj(a, b, d, e, f, g, h);
  } catch (l) {
    S(k);
    if (l !== l + 0 && "longjmp" !== l) {
      throw l;
    }
    X(1, 0);
  }
}
Object.getOwnPropertyDescriptor(u, "intArrayFromString") || (u.intArrayFromString = () => v("'intArrayFromString' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "intArrayToString") || (u.intArrayToString = () => v("'intArrayToString' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "ccall") || (u.ccall = () => v("'ccall' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "cwrap") || (u.cwrap = () => v("'cwrap' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "setValue") || (u.setValue = () => v("'setValue' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "getValue") || (u.getValue = () => v("'getValue' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "allocate") || (u.allocate = () => v("'allocate' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "UTF8ArrayToString") || (u.UTF8ArrayToString = () => v("'UTF8ArrayToString' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
u.UTF8ToString = db;
Object.getOwnPropertyDescriptor(u, "stringToUTF8Array") || (u.stringToUTF8Array = () => v("'stringToUTF8Array' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
u.stringToUTF8 = fb;
u.lengthBytesUTF8 = gb;
Object.getOwnPropertyDescriptor(u, "stackTrace") || (u.stackTrace = () => v("'stackTrace' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "addOnPreRun") || (u.addOnPreRun = () => v("'addOnPreRun' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "addOnInit") || (u.addOnInit = () => v("'addOnInit' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "addOnPreMain") || (u.addOnPreMain = () => v("'addOnPreMain' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "addOnExit") || (u.addOnExit = () => v("'addOnExit' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "addOnPostRun") || (u.addOnPostRun = () => v("'addOnPostRun' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "writeStringToMemory") || (u.writeStringToMemory = () => v("'writeStringToMemory' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "writeArrayToMemory") || (u.writeArrayToMemory = () => v("'writeArrayToMemory' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "writeAsciiToMemory") || (u.writeAsciiToMemory = () => v("'writeAsciiToMemory' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "addRunDependency") || (u.addRunDependency = () => v("'addRunDependency' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"));
Object.getOwnPropertyDescriptor(u, "removeRunDependency") || (u.removeRunDependency = () => v("'removeRunDependency' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"));
Object.getOwnPropertyDescriptor(u, "FS_createFolder") || (u.FS_createFolder = () => v("'FS_createFolder' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "FS_createPath") || (u.FS_createPath = () => v("'FS_createPath' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"));
Object.getOwnPropertyDescriptor(u, "FS_createDataFile") || (u.FS_createDataFile = () => v("'FS_createDataFile' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"));
Object.getOwnPropertyDescriptor(u, "FS_createPreloadedFile") || (u.FS_createPreloadedFile = () => v("'FS_createPreloadedFile' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"));
Object.getOwnPropertyDescriptor(u, "FS_createLazyFile") || (u.FS_createLazyFile = () => v("'FS_createLazyFile' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"));
Object.getOwnPropertyDescriptor(u, "FS_createLink") || (u.FS_createLink = () => v("'FS_createLink' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "FS_createDevice") || (u.FS_createDevice = () => v("'FS_createDevice' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"));
Object.getOwnPropertyDescriptor(u, "FS_unlink") || (u.FS_unlink = () => v("'FS_unlink' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ). Alternatively, forcing filesystem support (-s FORCE_FILESYSTEM=1) can export this for you"));
Object.getOwnPropertyDescriptor(u, "getLEB") || (u.getLEB = () => v("'getLEB' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "getFunctionTables") || (u.getFunctionTables = () => v("'getFunctionTables' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "alignFunctionTables") || (u.alignFunctionTables = () => v("'alignFunctionTables' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerFunctions") || (u.registerFunctions = () => v("'registerFunctions' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "addFunction") || (u.addFunction = () => v("'addFunction' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "removeFunction") || (u.removeFunction = () => v("'removeFunction' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "getFuncWrapper") || (u.getFuncWrapper = () => v("'getFuncWrapper' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "prettyPrint") || (u.prettyPrint = () => v("'prettyPrint' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "dynCall") || (u.dynCall = () => v("'dynCall' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "getCompilerSetting") || (u.getCompilerSetting = () => v("'getCompilerSetting' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "print") || (u.print = () => v("'print' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "printErr") || (u.printErr = () => v("'printErr' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "getTempRet0") || (u.getTempRet0 = () => v("'getTempRet0' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "setTempRet0") || (u.setTempRet0 = () => v("'setTempRet0' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "callMain") || (u.callMain = () => v("'callMain' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "abort") || (u.abort = () => v("'abort' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
u.keepRuntimeAlive = Ja;
Object.getOwnPropertyDescriptor(u, "zeroMemory") || (u.zeroMemory = () => v("'zeroMemory' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "stringToNewUTF8") || (u.stringToNewUTF8 = () => v("'stringToNewUTF8' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "setFileTime") || (u.setFileTime = () => v("'setFileTime' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "emscripten_realloc_buffer") || (u.emscripten_realloc_buffer = () => v("'emscripten_realloc_buffer' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "ENV") || (u.ENV = () => v("'ENV' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "withStackSave") || (u.withStackSave = () => v("'withStackSave' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "ERRNO_CODES") || (u.ERRNO_CODES = () => v("'ERRNO_CODES' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "ERRNO_MESSAGES") || (u.ERRNO_MESSAGES = () => v("'ERRNO_MESSAGES' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "setErrNo") || (u.setErrNo = () => v("'setErrNo' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "inetPton4") || (u.inetPton4 = () => v("'inetPton4' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "inetNtop4") || (u.inetNtop4 = () => v("'inetNtop4' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "inetPton6") || (u.inetPton6 = () => v("'inetPton6' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "inetNtop6") || (u.inetNtop6 = () => v("'inetNtop6' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "readSockaddr") || (u.readSockaddr = () => v("'readSockaddr' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "writeSockaddr") || (u.writeSockaddr = () => v("'writeSockaddr' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "DNS") || (u.DNS = () => v("'DNS' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "getHostByName") || (u.getHostByName = () => v("'getHostByName' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "GAI_ERRNO_MESSAGES") || (u.GAI_ERRNO_MESSAGES = () => v("'GAI_ERRNO_MESSAGES' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "Protocols") || (u.Protocols = () => v("'Protocols' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "Sockets") || (u.Sockets = () => v("'Sockets' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "getRandomDevice") || (u.getRandomDevice = () => v("'getRandomDevice' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "traverseStack") || (u.traverseStack = () => v("'traverseStack' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "convertFrameToPC") || (u.convertFrameToPC = () => v("'convertFrameToPC' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "UNWIND_CACHE") || (u.UNWIND_CACHE = () => v("'UNWIND_CACHE' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "saveInUnwindCache") || (u.saveInUnwindCache = () => v("'saveInUnwindCache' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "convertPCtoSourceLocation") || (u.convertPCtoSourceLocation = () => v("'convertPCtoSourceLocation' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "readAsmConstArgsArray") || (u.readAsmConstArgsArray = () => v("'readAsmConstArgsArray' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "readAsmConstArgs") || (u.readAsmConstArgs = () => v("'readAsmConstArgs' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "mainThreadEM_ASM") || (u.mainThreadEM_ASM = () => v("'mainThreadEM_ASM' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "jstoi_q") || (u.jstoi_q = () => v("'jstoi_q' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "jstoi_s") || (u.jstoi_s = () => v("'jstoi_s' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "getExecutableName") || (u.getExecutableName = () => v("'getExecutableName' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "listenOnce") || (u.listenOnce = () => v("'listenOnce' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "autoResumeAudioContext") || (u.autoResumeAudioContext = () => v("'autoResumeAudioContext' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "dynCallLegacy") || (u.dynCallLegacy = () => v("'dynCallLegacy' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "getDynCaller") || (u.getDynCaller = () => v("'getDynCaller' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "dynCall") || (u.dynCall = () => v("'dynCall' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "callRuntimeCallbacks") || (u.callRuntimeCallbacks = () => v("'callRuntimeCallbacks' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "wasmTableMirror") || (u.wasmTableMirror = () => v("'wasmTableMirror' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "setWasmTableEntry") || (u.setWasmTableEntry = () => v("'setWasmTableEntry' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "getWasmTableEntry") || (u.getWasmTableEntry = () => v("'getWasmTableEntry' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "handleException") || (u.handleException = () => v("'handleException' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "runtimeKeepalivePush") || (u.runtimeKeepalivePush = () => v("'runtimeKeepalivePush' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "runtimeKeepalivePop") || (u.runtimeKeepalivePop = () => v("'runtimeKeepalivePop' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "callUserCallback") || (u.callUserCallback = () => v("'callUserCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "maybeExit") || (u.maybeExit = () => v("'maybeExit' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "safeSetTimeout") || (u.safeSetTimeout = () => v("'safeSetTimeout' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "asmjsMangle") || (u.asmjsMangle = () => v("'asmjsMangle' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "asyncLoad") || (u.asyncLoad = () => v("'asyncLoad' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "alignMemory") || (u.alignMemory = () => v("'alignMemory' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "mmapAlloc") || (u.mmapAlloc = () => v("'mmapAlloc' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "reallyNegative") || (u.reallyNegative = () => v("'reallyNegative' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "unSign") || (u.unSign = () => v("'unSign' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "reSign") || (u.reSign = () => v("'reSign' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "formatString") || (u.formatString = () => v("'formatString' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "PATH") || (u.PATH = () => v("'PATH' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "PATH_FS") || (u.PATH_FS = () => v("'PATH_FS' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "SYSCALLS") || (u.SYSCALLS = () => v("'SYSCALLS' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "syscallMmap2") || (u.syscallMmap2 = () => v("'syscallMmap2' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "syscallMunmap") || (u.syscallMunmap = () => v("'syscallMunmap' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "getSocketFromFD") || (u.getSocketFromFD = () => v("'getSocketFromFD' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "getSocketAddress") || (u.getSocketAddress = () => v("'getSocketAddress' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "JSEvents") || (u.JSEvents = () => v("'JSEvents' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerKeyEventCallback") || (u.registerKeyEventCallback = () => v("'registerKeyEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "specialHTMLTargets") || (u.specialHTMLTargets = () => v("'specialHTMLTargets' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "maybeCStringToJsString") || (u.maybeCStringToJsString = () => v("'maybeCStringToJsString' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "findEventTarget") || (u.findEventTarget = () => v("'findEventTarget' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "findCanvasEventTarget") || (u.findCanvasEventTarget = () => v("'findCanvasEventTarget' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "getBoundingClientRect") || (u.getBoundingClientRect = () => v("'getBoundingClientRect' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "fillMouseEventData") || (u.fillMouseEventData = () => v("'fillMouseEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerMouseEventCallback") || (u.registerMouseEventCallback = () => v("'registerMouseEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerWheelEventCallback") || (u.registerWheelEventCallback = () => v("'registerWheelEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerUiEventCallback") || (u.registerUiEventCallback = () => v("'registerUiEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerFocusEventCallback") || (u.registerFocusEventCallback = () => v("'registerFocusEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "fillDeviceOrientationEventData") || (u.fillDeviceOrientationEventData = () => v("'fillDeviceOrientationEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerDeviceOrientationEventCallback") || (u.registerDeviceOrientationEventCallback = () => v("'registerDeviceOrientationEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "fillDeviceMotionEventData") || (u.fillDeviceMotionEventData = () => v("'fillDeviceMotionEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerDeviceMotionEventCallback") || (u.registerDeviceMotionEventCallback = () => v("'registerDeviceMotionEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "screenOrientation") || (u.screenOrientation = () => v("'screenOrientation' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "fillOrientationChangeEventData") || (u.fillOrientationChangeEventData = () => v("'fillOrientationChangeEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerOrientationChangeEventCallback") || (u.registerOrientationChangeEventCallback = () => v("'registerOrientationChangeEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "fillFullscreenChangeEventData") || (u.fillFullscreenChangeEventData = () => v("'fillFullscreenChangeEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerFullscreenChangeEventCallback") || (u.registerFullscreenChangeEventCallback = () => v("'registerFullscreenChangeEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerRestoreOldStyle") || (u.registerRestoreOldStyle = () => v("'registerRestoreOldStyle' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "hideEverythingExceptGivenElement") || (u.hideEverythingExceptGivenElement = () => v("'hideEverythingExceptGivenElement' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "restoreHiddenElements") || (u.restoreHiddenElements = () => v("'restoreHiddenElements' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "setLetterbox") || (u.setLetterbox = () => v("'setLetterbox' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "currentFullscreenStrategy") || (u.currentFullscreenStrategy = () => v("'currentFullscreenStrategy' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "restoreOldWindowedStyle") || (u.restoreOldWindowedStyle = () => v("'restoreOldWindowedStyle' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "softFullscreenResizeWebGLRenderTarget") || (u.softFullscreenResizeWebGLRenderTarget = () => v("'softFullscreenResizeWebGLRenderTarget' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "doRequestFullscreen") || (u.doRequestFullscreen = () => v("'doRequestFullscreen' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "fillPointerlockChangeEventData") || (u.fillPointerlockChangeEventData = () => v("'fillPointerlockChangeEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerPointerlockChangeEventCallback") || (u.registerPointerlockChangeEventCallback = () => v("'registerPointerlockChangeEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerPointerlockErrorEventCallback") || (u.registerPointerlockErrorEventCallback = () => v("'registerPointerlockErrorEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "requestPointerLock") || (u.requestPointerLock = () => v("'requestPointerLock' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "fillVisibilityChangeEventData") || (u.fillVisibilityChangeEventData = () => v("'fillVisibilityChangeEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerVisibilityChangeEventCallback") || (u.registerVisibilityChangeEventCallback = () => v("'registerVisibilityChangeEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerTouchEventCallback") || (u.registerTouchEventCallback = () => v("'registerTouchEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "fillGamepadEventData") || (u.fillGamepadEventData = () => v("'fillGamepadEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerGamepadEventCallback") || (u.registerGamepadEventCallback = () => v("'registerGamepadEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerBeforeUnloadEventCallback") || (u.registerBeforeUnloadEventCallback = () => v("'registerBeforeUnloadEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "fillBatteryEventData") || (u.fillBatteryEventData = () => v("'fillBatteryEventData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "battery") || (u.battery = () => v("'battery' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerBatteryEventCallback") || (u.registerBatteryEventCallback = () => v("'registerBatteryEventCallback' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "setCanvasElementSize") || (u.setCanvasElementSize = () => v("'setCanvasElementSize' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "getCanvasElementSize") || (u.getCanvasElementSize = () => v("'getCanvasElementSize' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "demangle") || (u.demangle = () => v("'demangle' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "demangleAll") || (u.demangleAll = () => v("'demangleAll' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "jsStackTrace") || (u.jsStackTrace = () => v("'jsStackTrace' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "stackTrace") || (u.stackTrace = () => v("'stackTrace' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "getEnvStrings") || (u.getEnvStrings = () => v("'getEnvStrings' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "checkWasiClock") || (u.checkWasiClock = () => v("'checkWasiClock' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "flush_NO_FILESYSTEM") || (u.flush_NO_FILESYSTEM = () => v("'flush_NO_FILESYSTEM' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "writeI53ToI64") || (u.writeI53ToI64 = () => v("'writeI53ToI64' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "writeI53ToI64Clamped") || (u.writeI53ToI64Clamped = () => v("'writeI53ToI64Clamped' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "writeI53ToI64Signaling") || (u.writeI53ToI64Signaling = () => v("'writeI53ToI64Signaling' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "writeI53ToU64Clamped") || (u.writeI53ToU64Clamped = () => v("'writeI53ToU64Clamped' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "writeI53ToU64Signaling") || (u.writeI53ToU64Signaling = () => v("'writeI53ToU64Signaling' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "readI53FromI64") || (u.readI53FromI64 = () => v("'readI53FromI64' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "readI53FromU64") || (u.readI53FromU64 = () => v("'readI53FromU64' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "convertI32PairToI53") || (u.convertI32PairToI53 = () => v("'convertI32PairToI53' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "convertU32PairToI53") || (u.convertU32PairToI53 = () => v("'convertU32PairToI53' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "setImmediateWrapped") || (u.setImmediateWrapped = () => v("'setImmediateWrapped' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "clearImmediateWrapped") || (u.clearImmediateWrapped = () => v("'clearImmediateWrapped' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "polyfillSetImmediate") || (u.polyfillSetImmediate = () => v("'polyfillSetImmediate' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "uncaughtExceptionCount") || (u.uncaughtExceptionCount = () => v("'uncaughtExceptionCount' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "exceptionLast") || (u.exceptionLast = () => v("'exceptionLast' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "exceptionCaught") || (u.exceptionCaught = () => v("'exceptionCaught' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "ExceptionInfo") || (u.ExceptionInfo = () => v("'ExceptionInfo' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "CatchInfo") || (u.CatchInfo = () => v("'CatchInfo' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "exception_addRef") || (u.exception_addRef = () => v("'exception_addRef' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "exception_decRef") || (u.exception_decRef = () => v("'exception_decRef' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "Browser") || (u.Browser = () => v("'Browser' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "funcWrappers") || (u.funcWrappers = () => v("'funcWrappers' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "getFuncWrapper") || (u.getFuncWrapper = () => v("'getFuncWrapper' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "setMainLoop") || (u.setMainLoop = () => v("'setMainLoop' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "wget") || (u.wget = () => v("'wget' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "tempFixedLengthArray") || (u.tempFixedLengthArray = () => v("'tempFixedLengthArray' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "miniTempWebGLFloatBuffers") || (u.miniTempWebGLFloatBuffers = () => v("'miniTempWebGLFloatBuffers' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "heapObjectForWebGLType") || (u.heapObjectForWebGLType = () => v("'heapObjectForWebGLType' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "heapAccessShiftForWebGLHeap") || (u.heapAccessShiftForWebGLHeap = () => v("'heapAccessShiftForWebGLHeap' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "GL") || (u.GL = () => v("'GL' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "emscriptenWebGLGet") || (u.emscriptenWebGLGet = () => v("'emscriptenWebGLGet' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "computeUnpackAlignedImageSize") || (u.computeUnpackAlignedImageSize = () => v("'computeUnpackAlignedImageSize' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "emscriptenWebGLGetTexPixelData") || (u.emscriptenWebGLGetTexPixelData = () => v("'emscriptenWebGLGetTexPixelData' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "emscriptenWebGLGetUniform") || (u.emscriptenWebGLGetUniform = () => v("'emscriptenWebGLGetUniform' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "webglGetUniformLocation") || (u.webglGetUniformLocation = () => v("'webglGetUniformLocation' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "webglPrepareUniformLocationsBeforeFirstUse") || (u.webglPrepareUniformLocationsBeforeFirstUse = () => v("'webglPrepareUniformLocationsBeforeFirstUse' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "webglGetLeftBracePos") || (u.webglGetLeftBracePos = () => v("'webglGetLeftBracePos' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "emscriptenWebGLGetVertexAttrib") || (u.emscriptenWebGLGetVertexAttrib = () => v("'emscriptenWebGLGetVertexAttrib' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "writeGLArray") || (u.writeGLArray = () => v("'writeGLArray' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "AL") || (u.AL = () => v("'AL' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "SDL_unicode") || (u.SDL_unicode = () => v("'SDL_unicode' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "SDL_ttfContext") || (u.SDL_ttfContext = () => v("'SDL_ttfContext' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "SDL_audio") || (u.SDL_audio = () => v("'SDL_audio' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "SDL") || (u.SDL = () => v("'SDL' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "SDL_gfx") || (u.SDL_gfx = () => v("'SDL_gfx' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "GLUT") || (u.GLUT = () => v("'GLUT' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "EGL") || (u.EGL = () => v("'EGL' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "GLFW_Window") || (u.GLFW_Window = () => v("'GLFW_Window' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "GLFW") || (u.GLFW = () => v("'GLFW' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "GLEW") || (u.GLEW = () => v("'GLEW' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "IDBStore") || (u.IDBStore = () => v("'IDBStore' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "runAndAbortIfError") || (u.runAndAbortIfError = () => v("'runAndAbortIfError' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
u.PThread = T;
Object.getOwnPropertyDescriptor(u, "killThread") || (u.killThread = () => v("'killThread' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "cleanupThread") || (u.cleanupThread = () => v("'cleanupThread' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "registerTlsInit") || (u.registerTlsInit = () => v("'registerTlsInit' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "cancelThread") || (u.cancelThread = () => v("'cancelThread' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "spawnThread") || (u.spawnThread = () => v("'spawnThread' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "exitOnMainThread") || (u.exitOnMainThread = () => v("'exitOnMainThread' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "establishStackSpace") || (u.establishStackSpace = () => v("'establishStackSpace' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "invokeEntryPoint") || (u.invokeEntryPoint = () => v("'invokeEntryPoint' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "warnOnce") || (u.warnOnce = () => v("'warnOnce' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
u.stackSave = O;
u.stackRestore = S;
u.stackAlloc = Ac;
Object.getOwnPropertyDescriptor(u, "AsciiToString") || (u.AsciiToString = () => v("'AsciiToString' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "stringToAscii") || (u.stringToAscii = () => v("'stringToAscii' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "UTF16ToString") || (u.UTF16ToString = () => v("'UTF16ToString' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "stringToUTF16") || (u.stringToUTF16 = () => v("'stringToUTF16' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "lengthBytesUTF16") || (u.lengthBytesUTF16 = () => v("'lengthBytesUTF16' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "UTF32ToString") || (u.UTF32ToString = () => v("'UTF32ToString' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "stringToUTF32") || (u.stringToUTF32 = () => v("'stringToUTF32' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "lengthBytesUTF32") || (u.lengthBytesUTF32 = () => v("'lengthBytesUTF32' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "allocateUTF8") || (u.allocateUTF8 = () => v("'allocateUTF8' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
Object.getOwnPropertyDescriptor(u, "allocateUTF8OnStack") || (u.allocateUTF8OnStack = () => v("'allocateUTF8OnStack' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)"));
u.writeStackCookie = mb;
u.checkStackCookie = ob;
u.PThread = T;
u.wasmMemory = c;
u.ExitStatus = Fa;
Object.getOwnPropertyDescriptor(u, "ALLOC_NORMAL") || Object.defineProperty(u, "ALLOC_NORMAL", {configurable:!0, get:function() {
  v("'ALLOC_NORMAL' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)");
}});
Object.getOwnPropertyDescriptor(u, "ALLOC_STACK") || Object.defineProperty(u, "ALLOC_STACK", {configurable:!0, get:function() {
  v("'ALLOC_STACK' was not exported. add it to EXPORTED_RUNTIME_METHODS (see the FAQ)");
}});
var xj;
function Fa(a) {
  this.name = "ExitStatus";
  this.message = "Program terminated with exit(" + a + ")";
  this.status = a;
}
Bb = function yj() {
  xj || zj();
  xj || (Bb = yj);
};
function zj() {
  function a() {
    if (!xj && (xj = !0, u.calledRun = !0, !$a)) {
      wb();
      sa(u);
      if (u.onRuntimeInitialized) {
        u.onRuntimeInitialized();
      }
      assert(!u._main, 'compiled without a main, but one is present. if you added it from JS, use Module["onRuntimeInitialized"]');
      ob();
      if (!B) {
        if (u.postRun) {
          for ("function" == typeof u.postRun && (u.postRun = [u.postRun]); u.postRun.length;) {
            var b = u.postRun.shift();
            ub.unshift(b);
          }
        }
        xb(ub);
      }
    }
  }
  if (!(0 < zb)) {
    if (kh(), mb(), B) {
      sa(u), wb(), postMessage({cmd:"loaded"});
    } else {
      assert(!B);
      if (u.preRun) {
        for ("function" == typeof u.preRun && (u.preRun = [u.preRun]); u.preRun.length;) {
          yb();
        }
      }
      xb(rb);
      0 < zb || (u.setStatus ? (u.setStatus("Running..."), setTimeout(function() {
        setTimeout(function() {
          u.setStatus("");
        }, 1);
        a();
      }, 1)) : a(), ob());
    }
  }
}
u.run = zj;
function Sb(a) {
  if (B) {
    throw Rb(a), "unwind";
  }
  if (Ja()) {
    var b = "program exited (with status: " + a + "), but keepRuntimeAlive() is set (counter=" + vb + ") due to an async operation, so halting execution but not exiting the runtime or preventing further async execution (you can use emscripten_force_exit, if you want to force a true shutdown)";
    ta(b);
    E(b);
  } else {
    ob(), B || (jh(), xb(tb), fc[1].length && gc(1, 10), fc[2].length && gc(2, 10), T.$(), Wa = !0);
  }
  if (!Ja()) {
    T.$();
    if (u.onExit) {
      u.onExit(a);
    }
    $a = !0;
  }
  wa(a, new Fa(a));
}
if (u.preInit) {
  for ("function" == typeof u.preInit && (u.preInit = [u.preInit]); 0 < u.preInit.length;) {
    u.preInit.pop()();
  }
}
B && (noExitRuntime = !1, T.qa());
zj();



  return ortWasmSimdThreaded.ready
}
);
})();
if (typeof exports === 'object' && typeof module === 'object')
  module.exports = ortWasmSimdThreaded;
else if (typeof define === 'function' && define['amd'])
  define([], function() { return ortWasmSimdThreaded; });
else if (typeof exports === 'object')
  exports["ortWasmSimdThreaded"] = ortWasmSimdThreaded;
