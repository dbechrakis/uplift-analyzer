(function (root) {
  function normalQuantile(p) {
    if (p <= 0 || p >= 1) throw new RangeError("p must be strictly between 0 and 1");
    var a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
    var b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    var c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    var d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    var low = 0.02425, q, r;
    if (p < low) {
      q = Math.sqrt(-2 * Math.log(p));
      return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p > 1 - low) {
      q = Math.sqrt(-2 * Math.log(1 - p));
      return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  function totalN(p0, p1, alpha, power, share) {
    var zAlpha = normalQuantile(1 - alpha / 2);
    var zPower = normalQuantile(power);
    var pooled = share * p1 + (1 - share) * p0;
    var nullSd = Math.sqrt(pooled * (1 - pooled) * (1 / share + 1 / (1 - share)));
    var altSd = Math.sqrt(p1 * (1 - p1) / share + p0 * (1 - p0) / (1 - share));
    return Math.pow(zAlpha * nullSd + zPower * altSd, 2) / Math.pow(p1 - p0, 2);
  }

  function requiredSampleSize(baseline, relativeLift, alpha, power, share) {
    var p1 = baseline * (1 + relativeLift);
    if (!(p1 > 0 && p1 < 1)) throw new RangeError("treated rate must be strictly between 0 and 1");
    var n = totalN(baseline, p1, alpha, power, share);
    var nt = Math.ceil(n * share);
    var nc = Math.ceil(n * (1 - share));
    return { nTreated: nt, nControl: nc, nTotal: nt + nc, treatedRate: p1 };
  }

  function minimumDetectableEffect(baseline, nTotal, alpha, power, share) {
    var lo = 1e-12;
    var hi = (1 - baseline) * 0.999999;
    if (totalN(baseline, baseline + hi, alpha, power, share) > nTotal) throw new RangeError("sample too small");
    for (var i = 0; i < 200; i++) {
      var mid = (lo + hi) / 2;
      if (totalN(baseline, baseline + mid, alpha, power, share) > nTotal) lo = mid; else hi = mid;
    }
    var delta = (lo + hi) / 2;
    return { absoluteEffect: delta, relativeEffect: delta / baseline };
  }

  var api = { normalQuantile: normalQuantile, requiredSampleSize: requiredSampleSize, minimumDetectableEffect: minimumDetectableEffect };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Power = api;
})(typeof window !== "undefined" ? window : globalThis);
