(function (root) {
  function series(data, looks) {
    var key = String(looks);
    if (data.looks.indexOf(looks) < 0) throw new RangeError("looks must be one of " + data.looks.join(", "));
    return {
      rules: data.rules.slice(),
      falsePositive: data.rules.map(function (r) { return data.false_positive[r][key]; }),
      power: data.rules.map(function (r) { return data.power_at_5_percent_lift[r][key]; })
    };
  }

  var api = { series: series };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Peeking = api;
})(typeof window !== "undefined" ? window : globalThis);
