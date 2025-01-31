import type { PlotConfig } from "asciichart";

import * as asciichart from "asciichart";
import stripAnsi from "strip-ansi";

export interface Point {
  x: string;
  y: number;
}

function normalizeSeries(series: Point[][] | Point[]): Point[][] {
  if (!Array.isArray(series[0])) {
    return [series as Point[]];
  }

  return series as Point[][];
}

function extractYValues(series: Point[][]): number[][] {
  return series.map((points) => points.map((point) => point.y));
}

interface PlotConfigWithLineLabels extends PlotConfig {
  title?: string;
  yLabel?: string;
  lineLabels?: string[];
  xLabel?: string;
  width?: number;
}

export function plot(series: Point[][] | Point[], config: PlotConfigWithLineLabels = {}): string {
  // Normalize input to MultiSeries
  const normalizedSeries = normalizeSeries(series);
  // Convert to arrays of y-values for plotting
  let yArrays = extractYValues(normalizedSeries);
  // Get x-values from the first series
  const xArray = normalizedSeries[0]!.map((point) => point.x);
  // Validate non-empty arrays
  yArrays.forEach((arr) => {
    if (arr.length === 0) throw new Error("Cannot plot empty array");
  });

  if (config.width) {
    yArrays = yArrays.map((arr) => {
      const newArr = [];
      for (let i = 0; i < config.width!; i++) {
        newArr.push(arr[Math.floor((i * arr.length) / config.width!)]);
      }

      const isFull = newArr.every((element) => element && element !== undefined);

      if (isFull) {
        return newArr as number[];
      }

      return arr;
    });
  }

  const plot = asciichart.plot(yArrays, config);
  // determine the overall width of the plot (in characters)
  const plotFirstLine = stripAnsi(plot).split("\n")[0];
  if (!plotFirstLine || plotFirstLine.length === 0) {
    throw new Error("Empty plot");
  }
  const fullWidth = plotFirstLine.length;
  // get the number of characters reserved for the y-axis legend
  const leftMargin = plotFirstLine.split(/┤|┼╮|┼/)[0]?.length ? plotFirstLine.split(/┤|┼╮|┼/)[0]!.length + 1 : 0;
  // the difference between the two is the actual width of the x axis
  const widthXaxis = fullWidth - leftMargin;

  // Calculate tick distance based on number of points
  const tickDistance = Math.floor(widthXaxis / xArray.length);

  // Generate ticks
  let ticks = " ".repeat(leftMargin - 1);
  xArray.map(() => {
    ticks += "┬" + "─".repeat(tickDistance - 1);
  });

  if (ticks.length < fullWidth) {
    ticks += "─".repeat(fullWidth - ticks.length);
  }

  // Generate tick labels
  let tickLabels = " ".repeat(leftMargin - 1);
  // eslint-disable-next-line @typescript-eslint/prefer-for-of
  for (let i = 0; i < xArray.length; i++) {
    tickLabels += xArray[i]!.toString()
      .substring(0, tickDistance - 1)
      .padEnd(tickDistance);
  }
  if (tickLabels.length < fullWidth) {
    tickLabels += " ".repeat(fullWidth - tickLabels.length);
  }

  const title =
    config.title ?
      `${" ".repeat(Math.floor(Math.abs(leftMargin + (widthXaxis - config.title.length) / 2)))}${config.title}\n\n`
    : "";
  let yLabel = "";
  if (config.yLabel || config.lineLabels) {
    if (config.yLabel) {
      yLabel += `${asciichart.darkgray}${config.yLabel.padStart(leftMargin + config.yLabel.length / 2)}${asciichart.reset}`;
    }
    if (config.lineLabels) {
      let legend = "";
      for (let i = 0; i < Math.min(yArrays.length, config.lineLabels.length); i++) {
        const color = Array.isArray(config.colors) ? config.colors[i] : asciichart.default;
        legend += `    ${color}─── ${config.lineLabels[i]}${asciichart.reset}`;
      }
      yLabel +=
        " ".repeat(Math.floor(Math.abs(fullWidth - 1 - stripAnsi(legend).length - stripAnsi(yLabel).length))) + legend;
    }
    yLabel += `\n${"╷".padStart(leftMargin)}\n`;
  }
  const xLabel =
    config.xLabel ? `\n${asciichart.darkgray}${config.xLabel.padStart(fullWidth - 1)}${asciichart.reset}` : "";
  return `\n${title}${yLabel}${plot}\n${ticks}\n${tickLabels}${xLabel}\n`;
}
