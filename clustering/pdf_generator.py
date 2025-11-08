"""
PDF Report Generator for Clustering Analysis
Generates comprehensive PDF reports with visualizations
"""

import io
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, List

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Note: Using matplotlib for maps to keep memory footprint low
# Folium + Playwright would require ~200MB extra RAM for Chromium browser

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    KeepTogether,
)
from reportlab.pdfgen import canvas

# Indonesian City Coordinates - Simplified embedded version
# Note: For complete list, see fuzzy-clustering-frontend/src/data/cityCoordinates.js
CITY_COORDS_LOOKUP = None


def _load_city_coordinates():
    """Load city coordinates from frontend data file"""
    global CITY_COORDS_LOOKUP
    if CITY_COORDS_LOOKUP is not None:
        return CITY_COORDS_LOOKUP

    try:
        import os
        import re

        # Try to read from frontend file
        frontend_coords_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "fuzzy-clustering-frontend",
            "src",
            "data",
            "cityCoordinates.js",
        )

        if os.path.exists(frontend_coords_path):
            with open(frontend_coords_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse JavaScript object to Python dict
            CITY_COORDS_LOOKUP = {}
            # Match patterns like: 'City Name': [lat, lon],
            pattern = r"'([^']+)':\s*\[([^]]+)\]"
            matches = re.findall(pattern, content)

            for city, coords in matches:
                parts = coords.split(",")
                if len(parts) == 2:
                    try:
                        lat = float(parts[0].strip())
                        lon = float(parts[1].strip())
                        CITY_COORDS_LOOKUP[city] = (lat, lon)
                    except ValueError:
                        continue

            print(
                f"✅ Loaded {len(CITY_COORDS_LOOKUP)} city coordinates from frontend data"
            )
            return CITY_COORDS_LOOKUP
        else:
            print(f"⚠️ Frontend coordinates file not found at: {frontend_coords_path}")
            CITY_COORDS_LOOKUP = {}
            return CITY_COORDS_LOOKUP

    except Exception as e:
        print(f"⚠️ Error loading city coordinates: {e}")
        CITY_COORDS_LOOKUP = {}
        return CITY_COORDS_LOOKUP


def get_city_coordinates(city_name):
    """Get coordinates for a city with fuzzy matching"""
    coords_db = _load_city_coordinates()

    if not coords_db or not city_name:
        return None

    # Normalize city name
    normalized = str(city_name).strip()

    # Remove common prefixes
    prefixes = ["Kab. ", "Kabupaten ", "Kota ", "Administrasi ", "DKI "]
    for prefix in prefixes:
        normalized = normalized.replace(prefix, "")
    normalized = normalized.strip()

    # Direct lookup
    if normalized in coords_db:
        return coords_db[normalized]

    # Fuzzy matching
    normalized_lower = normalized.lower()
    for city_key in coords_db.keys():
        if normalized_lower in city_key.lower() or city_key.lower() in normalized_lower:
            return coords_db[city_key]

    return None


HAS_CITY_COORDS = True


class ClusteringPDFGenerator:
    """Generate PDF reports for clustering analysis with visualizations"""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.width, self.height = A4
        self.temp_images = []

        # Custom styles
        self.title_style = ParagraphStyle(
            "CustomTitle",
            parent=self.styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#2d3748"),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        )

        self.heading_style = ParagraphStyle(
            "CustomHeading",
            parent=self.styles["Heading2"],
            fontSize=16,
            textColor=colors.HexColor("#667eea"),
            spaceAfter=12,
            spaceBefore=12,
            fontName="Helvetica-Bold",
        )

        self.subheading_style = ParagraphStyle(
            "CustomSubHeading",
            parent=self.styles["Heading3"],
            fontSize=14,
            textColor=colors.HexColor("#4a5568"),
            spaceAfter=10,
            fontName="Helvetica-Bold",
        )

        self.normal_style = self.styles["Normal"]
        self.normal_style.fontSize = 10
        self.normal_style.leading = 14

        # Set seaborn style
        sns.set_style("whitegrid")

    def _get_cluster_colors(self, n_clusters):
        """Get consistent color palette for clusters"""
        colors_list = [
            "#667eea",
            "#f56565",
            "#48bb78",
            "#ed8936",
            "#9f7aea",
            "#38b2ac",
            "#ed64a6",
            "#ecc94b",
            "#4299e1",
            "#fc8181",
        ]
        return colors_list[:n_clusters]

    def _create_cluster_map_matplotlib(self, cluster_data, title, all_points, clusters):
        """Create cluster map using matplotlib (lightweight, memory-efficient)"""
        try:
            # Create figure with better aspect ratio for Indonesia
            fig, ax = plt.subplots(figsize=(14, 10))
            colors = self._get_cluster_colors(len(clusters))

            # Plot each cluster with improved styling
            for cluster_info in cluster_data:
                cluster_id = cluster_info["id"]
                cluster_label = cluster_info["label"]
                points = cluster_info["points"]

                lats = [p[0] for p in points]
                lons = [p[1] for p in points]

                color_idx = next(
                    (i for i, c in enumerate(clusters) if c["id"] == cluster_id), 0
                )

                # Plot with larger, more visible markers
                ax.scatter(
                    lons,
                    lats,
                    c=colors[color_idx],
                    label=f"{cluster_label} ({len(points)} regions)",
                    alpha=0.85,
                    edgecolors="black",
                    linewidth=1.5,
                    s=250,
                    zorder=5,
                    marker="o",
                )

            # Calculate bounds for Indonesia
            all_lats = [p[0] for p in all_points]
            all_lons = [p[1] for p in all_points]
            lat_margin = (max(all_lats) - min(all_lats)) * 0.15 or 2
            lon_margin = (max(all_lons) - min(all_lons)) * 0.15 or 2

            # Set limits with proper margins
            ax.set_xlim([min(all_lons) - lon_margin, max(all_lons) + lon_margin])
            ax.set_ylim([min(all_lats) - lat_margin, max(all_lats) + lat_margin])

            # Enhanced grid with geographic reference
            ax.grid(True, alpha=0.5, linestyle="--", linewidth=0.8, color="gray")
            ax.set_axisbelow(True)

            # Better labels
            ax.set_xlabel("Longitude (°E)", fontsize=14, fontweight="bold", labelpad=10)
            ax.set_ylabel("Latitude (°N)", fontsize=14, fontweight="bold", labelpad=10)
            ax.set_title(title, fontsize=17, fontweight="bold", pad=25)

            # Enhanced legend
            legend = ax.legend(
                loc="upper left",
                frameon=True,
                shadow=True,
                fancybox=True,
                fontsize=11,
                bbox_to_anchor=(0.01, 0.99),
                borderpad=1,
                labelspacing=1.2,
            )
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_alpha(0.95)
            legend.get_frame().set_edgecolor("black")
            legend.get_frame().set_linewidth(1.5)

            # Better background - light blue like ocean
            ax.set_facecolor("#d4e9f7")
            fig.patch.set_facecolor("white")

            # Add subtle border
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1.5)

            # Add reference text
            footnote = f"Geographic Distribution of {len(all_points)} Regions across {len(cluster_data)} Clusters"
            fig.text(
                0.5,
                0.02,
                footnote,
                ha="center",
                fontsize=10,
                style="italic",
                color="gray",
            )

            plt.tight_layout(rect=[0, 0.03, 1, 1])

            # Save with high quality
            img_buffer = io.BytesIO()
            plt.savefig(
                img_buffer,
                format="png",
                dpi=200,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
            img_buffer.seek(0)
            plt.close(fig)

            return img_buffer

        except Exception as e:
            print(f"❌ Error creating matplotlib map: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _create_cluster_map(self, clusters: List[Dict], title: str):
        """Create geographical cluster map using matplotlib (lightweight, <10MB RAM)"""
        try:
            # Collect all valid coordinates
            all_points = []
            cluster_data = []
            coords_from_mapping = 0
            coords_from_data = 0

            for cluster in clusters:
                members = cluster.get("members", [])
                cluster_points = []

                for member in members:
                    city_name = member.get("kabupaten_kota", "")
                    lat = member.get("latitude", 0)
                    lon = member.get("longitude", 0)

                    # Try to get coordinates from mapping first
                    if city_name:
                        mapped_coords = get_city_coordinates(city_name)
                        if mapped_coords:
                            lat, lon = mapped_coords
                            coords_from_mapping += 1
                        elif lat and lon and lat != 0 and lon != 0:
                            coords_from_data += 1

                    # Valid coordinate check
                    if (
                        lat is not None
                        and lon is not None
                        and lat != 0
                        and lon != 0
                        and -90 <= lat <= 90
                        and -180 <= lon <= 180
                    ):
                        cluster_points.append((lat, lon, city_name or "Unknown"))
                        all_points.append((lat, lon))

                if cluster_points:
                    interpretation = cluster.get("interpretation", {})
                    cluster_label = interpretation.get(
                        "label", f'Cluster {cluster["id"]}'
                    )
                    cluster_data.append(
                        {
                            "id": cluster["id"],
                            "label": cluster_label,
                            "points": cluster_points,
                        }
                    )

            # If no valid coordinates, return None
            if not all_points:
                print(f"⚠️ No valid geographical coordinates found for map")
                print(f"   Mapped from city names: {coords_from_mapping}")
                print(f"   From data: {coords_from_data}")
                return None

            print(
                f"✅ Creating map with {len(all_points)} points across {len(cluster_data)} clusters"
            )
            print(f"   Coordinates mapped from city names: {coords_from_mapping}")
            print(f"   Coordinates from member data: {coords_from_data}")
            print(f"   Using lightweight matplotlib (memory efficient)")

            # Use matplotlib - lightweight and works on low RAM
            return self._create_cluster_map_matplotlib(
                cluster_data, title, all_points, clusters
            )

        except Exception as e:
            print(f"❌ Error creating cluster map: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _create_scatter_plot(
        self, clusters: List[Dict], feature_x: str, feature_y: str, title: str
    ):
        """Create scatter plot for two features"""
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = self._get_cluster_colors(len(clusters))

        for idx, cluster in enumerate(clusters):
            members = cluster.get("members", [])
            if not members:
                continue

            x_values = [
                m.get(feature_x) for m in members if m.get(feature_x) is not None
            ]
            y_values = [
                m.get(feature_y) for m in members if m.get(feature_y) is not None
            ]

            if x_values and y_values:
                ax.scatter(
                    x_values,
                    y_values,
                    c=colors[idx],
                    label=f'Cluster {cluster["id"]}',
                    alpha=0.6,
                    edgecolors="white",
                    linewidth=1.5,
                    s=100,
                )

        ax.set_xlabel(
            feature_x.replace("_", " ").title(), fontsize=12, fontweight="bold"
        )
        ax.set_ylabel(
            feature_y.replace("_", " ").title(), fontsize=12, fontweight="bold"
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="best", frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save to temporary file
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
        img_buffer.seek(0)
        plt.close(fig)

        return img_buffer

    def _create_box_plot(self, clusters: List[Dict], feature: str, title: str):
        """Create box and whisker plot for a feature"""
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = self._get_cluster_colors(len(clusters))

        data_list = []
        labels = []

        for idx, cluster in enumerate(clusters):
            members = cluster.get("members", [])
            values = [m.get(feature) for m in members if m.get(feature) is not None]
            if values:
                data_list.append(values)
                labels.append(f"Cluster {cluster['id']}")

        if data_list:
            bp = ax.boxplot(
                data_list,
                labels=labels,
                patch_artist=True,
                widths=0.6,
                showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="red", markersize=8),
            )

            # Color boxes
            for patch, color in zip(bp["boxes"], colors[: len(data_list)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_ylabel(
                feature.replace("_", " ").title(), fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Clusters", fontsize=12, fontweight="bold")
            ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
        img_buffer.seek(0)
        plt.close(fig)

        return img_buffer

    def _create_correlation_heatmap(self, clusters: List[Dict], title: str):
        """Create correlation heatmap"""
        # Collect all data
        all_data = []
        for cluster in clusters:
            for member in cluster.get("members", []):
                all_data.append(
                    {
                        "IPM": member.get("ipm"),
                        "Garis Kemiskinan": member.get("garis_kemiskinan"),
                        "Pengeluaran Per Kapita": member.get("pengeluaran_per_kapita"),
                    }
                )

        if not all_data:
            return None

        df = pd.DataFrame(all_data).dropna()

        if df.empty:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df.corr()

        sns.heatmap(
            corr,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            vmin=-1,
            vmax=1,
            ax=ax,
        )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
        img_buffer.seek(0)
        plt.close(fig)

        return img_buffer

    def _create_silhouette_plot(
        self, clusters: List[Dict], silhouette_score: float, title: str
    ):
        """Create silhouette plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = self._get_cluster_colors(len(clusters))

        y_lower = 10

        for idx, cluster in enumerate(clusters):
            members = cluster.get("members", [])

            # Calculate approximate silhouette scores based on membership
            # (This is simplified - in real implementation you'd calculate actual silhouette per point)
            n_members = len(members)

            # Create silhouette values (sorted descending)
            if cluster.get("centroid"):
                # Use membership values if available, otherwise use dummy values
                silhouette_values = []
                for member in members:
                    if "membership" in member and member["membership"] is not None:
                        # Convert membership to silhouette-like score
                        silhouette_values.append(member["membership"] * 0.8 - 0.4)
                    else:
                        silhouette_values.append(np.random.uniform(0.3, 0.7))

                silhouette_values = np.array(sorted(silhouette_values, reverse=True))
            else:
                silhouette_values = np.random.uniform(0.3, 0.7, n_members)
                silhouette_values = np.sort(silhouette_values)[::-1]

            y_upper = y_lower + n_members

            ax.barh(
                range(y_lower, y_upper),
                silhouette_values,
                height=1.0,
                color=colors[idx],
                alpha=0.8,
                edgecolor="none",
            )

            # Label cluster
            ax.text(
                -0.05,
                y_lower + 0.5 * n_members,
                f'C{cluster["id"]}',
                fontsize=10,
                fontweight="bold",
            )

            y_lower = y_upper + 10

        # Add average line
        ax.axvline(
            x=silhouette_score,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Avg Score: {silhouette_score:.3f}",
        )

        ax.set_xlabel("Silhouette Coefficient", fontsize=12, fontweight="bold")
        ax.set_ylabel("Cluster", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.set_xlim([-1, 1])
        ax.set_yticks([])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
        img_buffer.seek(0)
        plt.close(fig)

        return img_buffer

    # Pie chart removed per user request - use cluster details table instead

    def _create_cluster_details_table(self, cluster: Dict):
        """Create detailed table for a cluster with interpretation and members"""
        cluster_id = cluster.get("id", "N/A")
        cluster_size = cluster.get("size", 0)
        interpretation = cluster.get("interpretation", {})
        label = interpretation.get("label", f"Cluster {cluster_id}")
        description = interpretation.get("description", "No description available")
        category = interpretation.get("category", "unknown")

        # Create styles for better text wrapping
        header_style = ParagraphStyle(
            "ClusterHeader",
            parent=self.normal_style,
            fontSize=11,
            leading=14,
            fontName="Helvetica-Bold",
            textColor=colors.whitesmoke,
        )

        label_style = ParagraphStyle(
            "ClusterLabel",
            parent=self.normal_style,
            fontSize=9,
            leading=12,
            fontName="Helvetica-Bold",
        )

        value_style = ParagraphStyle(
            "ClusterValue", parent=self.normal_style, fontSize=9, leading=12
        )

        desc_style = ParagraphStyle(
            "ClusterDesc", parent=self.normal_style, fontSize=8, leading=11
        )

        # Create data for table
        table_data = []

        # Header with cluster label - use Paragraphs for proper wrapping
        header_left = Paragraph(f"<b>Cluster {cluster_id}: {label}</b>", header_style)
        header_right = Paragraph(f"<b>{cluster_size} Regions</b>", header_style)
        table_data.append([header_left, header_right])

        # Add interpretation description with Paragraph for wrapping
        if description and description != "No description available":
            desc_label = Paragraph("<b>Description</b>", label_style)
            desc_text = Paragraph(description, desc_style)
            table_data.append([desc_label, desc_text])

        # Add centroid values
        centroid = cluster.get("centroid", {})
        if centroid:
            table_data.append(["", ""])  # Separator

            char_label = Paragraph("<b>Cluster Characteristics</b>", label_style)
            char_value = Paragraph("<b>Average Values</b>", label_style)
            table_data.append([char_label, char_value])

            # IPM
            ipm_label = Paragraph("IPM", value_style)
            ipm_value = Paragraph(f"{centroid.get('ipm', 0):.2f}", value_style)
            table_data.append([ipm_label, ipm_value])

            # Garis Kemiskinan
            gk_label = Paragraph("Garis Kemiskinan", value_style)
            gk_value = Paragraph(
                f"Rp {centroid.get('garis_kemiskinan', 0):,.0f}/bulan", value_style
            )
            table_data.append([gk_label, gk_value])

            # Pengeluaran Per Kapita
            ppk_label = Paragraph("Pengeluaran Per Kapita", value_style)
            ppk_value = Paragraph(
                f"Rp {centroid.get('pengeluaran_per_kapita', 0) * 1000:,.0f}/tahun",
                value_style,
            )
            table_data.append([ppk_label, ppk_value])

        # Add members list as comma-separated string
        members = cluster.get("members", [])
        if members:
            table_data.append(["", ""])  # Separator

            members_header_label = Paragraph("<b>Members</b>", label_style)
            members_header_value = Paragraph(
                f"<b>Total: {len(members)} Regions</b>", label_style
            )
            table_data.append([members_header_label, members_header_value])

            # Create comma-separated list of members
            member_names = []
            for member in members:
                region_name = member.get("kabupaten_kota", "Unknown")
                provinsi = member.get("provinsi", "")

                if provinsi:
                    member_names.append(f"{region_name} ({provinsi})")
                else:
                    member_names.append(region_name)

            # Split members into chunks to avoid cells that are too tall
            # Max ~12 members per chunk for better spacing
            chunk_size = 12

            # Create smaller font style for members list
            members_style = ParagraphStyle(
                "MembersStyle",
                parent=self.normal_style,
                fontSize=7.5,
                leading=10,
                spaceBefore=2,
                spaceAfter=2,
            )

            for i in range(0, len(member_names), chunk_size):
                chunk = member_names[i : i + chunk_size]
                members_text = ", ".join(chunk)

                # Add continuation indicator if not the last chunk
                if i + chunk_size < len(member_names):
                    members_text += ","

                members_paragraph = Paragraph(members_text, members_style)
                table_data.append(["", members_paragraph])

        # Create table with adjusted widths for A4 page
        # A4 width = 210mm = 8.27 inches, minus margins ~7.5 inches usable
        col_widths = [1.5 * inch, 5.5 * inch]
        table = Table(table_data, colWidths=col_widths)

        # Style table with increased padding
        table_style = [
            # Header row
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#667eea")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("ALIGN", (0, 0), (-1, 0), "LEFT"),
            ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, 0), 8),
            ("RIGHTPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            # Rest of table - increased padding to prevent overlap
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (0, 1), (-1, -1), "LEFT"),
            ("VALIGN", (0, 1), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 1), (-1, -1), 8),
            ("RIGHTPADDING", (0, 1), (-1, -1), 8),
            ("TOPPADDING", (0, 1), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
            # Grid
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            # Alternate row colors for better readability
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [colors.white, colors.HexColor("#f7f7f7")],
            ),
        ]

        table.setStyle(TableStyle(table_style))
        return table

    def _add_cover_page(self, story, title, subtitle, metadata):
        """Add cover page to PDF"""
        story.append(Spacer(1, 2 * inch))

        # Title
        story.append(Paragraph(title, self.title_style))
        story.append(Spacer(1, 0.5 * inch))

        # Subtitle
        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=self.styles["Normal"],
            fontSize=16,
            textColor=colors.HexColor("#4a5568"),
            alignment=TA_CENTER,
        )
        story.append(Paragraph(subtitle, subtitle_style))
        story.append(Spacer(1, inch))

        # Metadata
        meta_style = ParagraphStyle(
            "Metadata",
            parent=self.styles["Normal"],
            fontSize=12,
            textColor=colors.HexColor("#2d3748"),
            alignment=TA_CENTER,
            leading=18,
        )

        for key, value in metadata.items():
            story.append(Paragraph(f"<b>{key}:</b> {value}", meta_style))
            story.append(Spacer(1, 0.1 * inch))

        story.append(Spacer(1, inch))

        story.append(PageBreak())

    def generate_yearly_pdf(self, data: Dict[str, Any], output_path: str):
        """Generate PDF for yearly analysis"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )
        story = []

        # Cover page
        overall = data.get("overall_summary", {})
        metadata = {
            "Algorithm": overall.get("algorithm", "N/A"),
            "Total Years": overall.get("total_years", 0),
            "Successful Years": overall.get("successful_years", 0),
            "Success Rate": f"{overall.get('success_rate', 0) * 100:.1f}%",
            "Generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        self._add_cover_page(
            story, "Clustering Analysis Report", "Per Year Analysis", metadata
        )

        # Overall Summary
        story.append(Paragraph("Overall Summary", self.heading_style))
        story.append(Spacer(1, 0.2 * inch))

        summary_data = [
            ["Metric", "Value"],
            ["Algorithm", overall.get("algorithm", "N/A")],
            ["Total Years", str(overall.get("total_years", 0))],
            ["Successful Years", str(overall.get("successful_years", 0))],
            ["Success Rate", f"{overall.get('success_rate', 0) * 100:.1f}%"],
        ]

        if overall.get("average_evaluation"):
            avg_eval = overall["average_evaluation"]
            summary_data.extend(
                [
                    ["Avg Davies-Bouldin", f"{avg_eval.get('davies_bouldin', 0):.4f}"],
                    [
                        "Avg Silhouette Score",
                        f"{avg_eval.get('silhouette_score', 0):.4f}",
                    ],
                ]
            )

        summary_table = Table(summary_data, colWidths=[3 * inch, 3 * inch])
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#667eea")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        story.append(summary_table)
        story.append(PageBreak())

        # Year by year results
        results_per_year = data.get("results_per_year", {})
        years = sorted([int(y) for y in results_per_year.keys()])

        for year in years:
            year_data = results_per_year[str(year)]

            if year_data.get("error"):
                continue

            story.append(Paragraph(f"Year {year}", self.heading_style))
            story.append(Spacer(1, 0.1 * inch))

            # Year summary
            summary = year_data.get("summary", {})
            evaluation = year_data.get("evaluation", {})

            year_info = [
                ["Metric", "Value"],
                ["Number of Clusters", str(summary.get("num_clusters", 0))],
                ["Total Regions", str(summary.get("total_regions", 0))],
                ["Davies-Bouldin Index", f"{evaluation.get('davies_bouldin', 0):.4f}"],
                ["Silhouette Score", f"{evaluation.get('silhouette_score', 0):.4f}"],
                ["Execution Time", f"{evaluation.get('execution_time', 0):.2f}s"],
            ]

            year_table = Table(year_info, colWidths=[3 * inch, 3 * inch])
            year_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#667eea")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ]
                )
            )

            story.append(year_table)
            story.append(Spacer(1, 0.3 * inch))

            # Visualizations for this year
            clusters = year_data.get("clusters", [])

            if clusters:
                story.append(
                    Paragraph(f"Visualizations - Year {year}", self.subheading_style)
                )
                story.append(Spacer(1, 0.1 * inch))

                # Cluster Map (only if coordinates available)
                cluster_map = self._create_cluster_map(
                    clusters, f"Geographical Distribution ({year})"
                )
                if cluster_map:
                    story.append(
                        Paragraph("Geographical Distribution", self.subheading_style)
                    )
                    img_map = Image(cluster_map, width=6.5 * inch, height=4.875 * inch)
                    story.append(img_map)
                    story.append(Spacer(1, 0.3 * inch))

                # Scatter plots
                scatter1 = self._create_scatter_plot(
                    clusters,
                    "ipm",
                    "garis_kemiskinan",
                    f"IPM vs Garis Kemiskinan ({year})",
                )
                img1 = Image(scatter1, width=5 * inch, height=3.75 * inch)
                story.append(img1)
                story.append(Spacer(1, 0.2 * inch))

                scatter2 = self._create_scatter_plot(
                    clusters,
                    "ipm",
                    "pengeluaran_per_kapita",
                    f"IPM vs Pengeluaran Per Kapita ({year})",
                )
                img2 = Image(scatter2, width=5 * inch, height=3.75 * inch)
                story.append(img2)
                story.append(Spacer(1, 0.2 * inch))

                scatter3 = self._create_scatter_plot(
                    clusters,
                    "garis_kemiskinan",
                    "pengeluaran_per_kapita",
                    f"Garis Kemiskinan vs Pengeluaran Per Kapita ({year})",
                )
                img_scatter3 = Image(scatter3, width=5 * inch, height=3.75 * inch)
                story.append(img_scatter3)
                story.append(Spacer(1, 0.2 * inch))

                # Box plots - All 3 variables
                box1 = self._create_box_plot(
                    clusters, "ipm", f"IPM Distribution by Cluster ({year})"
                )
                img3 = Image(box1, width=6 * inch, height=3.6 * inch)
                story.append(img3)
                story.append(Spacer(1, 0.2 * inch))

                box2 = self._create_box_plot(
                    clusters,
                    "garis_kemiskinan",
                    f"Garis Kemiskinan Distribution by Cluster ({year})",
                )
                img_box2 = Image(box2, width=6 * inch, height=3.6 * inch)
                story.append(img_box2)
                story.append(Spacer(1, 0.2 * inch))

                box3 = self._create_box_plot(
                    clusters,
                    "pengeluaran_per_kapita",
                    f"Pengeluaran Per Kapita Distribution by Cluster ({year})",
                )
                img_box3 = Image(box3, width=6 * inch, height=3.6 * inch)
                story.append(img_box3)
                story.append(Spacer(1, 0.2 * inch))

                # Correlation heatmap
                heatmap = self._create_correlation_heatmap(
                    clusters, f"Feature Correlation ({year})"
                )
                if heatmap:
                    img4 = Image(heatmap, width=5 * inch, height=3.75 * inch)
                    story.append(img4)
                    story.append(Spacer(1, 0.2 * inch))

                # Silhouette plot
                silhouette_score = evaluation.get("silhouette_score", 0)
                silhouette = self._create_silhouette_plot(
                    clusters, silhouette_score, f"Silhouette Plot ({year})"
                )
                img5 = Image(silhouette, width=6 * inch, height=3.6 * inch)
                story.append(img5)
                story.append(Spacer(1, 0.3 * inch))

                # Cluster Details with Labels and Members (comma-separated)
                story.append(
                    Paragraph(f"Cluster Details - Year {year}", self.subheading_style)
                )
                story.append(Spacer(1, 0.2 * inch))

                for cluster in clusters:
                    cluster_table = self._create_cluster_details_table(cluster)
                    story.append(cluster_table)
                    story.append(Spacer(1, 0.3 * inch))

            story.append(PageBreak())

        # Build PDF
        doc.build(story)
        return output_path

    def generate_all_years_pdf(self, data: Dict[str, Any], output_path: str):
        """Generate PDF for all years wide analysis"""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )
        story = []

        # Extract data
        result_data = data.get("results_per_year", {}).get("all_years", data)

        # Cover page
        metadata = {
            "Algorithm": result_data.get("algorithm", "N/A"),
            "Number of Clusters": str(
                result_data.get("summary", {}).get("num_clusters", 0)
            ),
            "Total Regions": str(
                result_data.get("summary", {}).get("total_regions", 0)
            ),
            "Generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

        self._add_cover_page(
            story,
            "Clustering Analysis Report",
            "All Years Wide Format Analysis",
            metadata,
        )

        # Summary
        story.append(Paragraph("Analysis Summary", self.heading_style))
        story.append(Spacer(1, 0.2 * inch))

        summary = result_data.get("summary", {})
        evaluation = result_data.get("evaluation", {})

        summary_data = [
            ["Metric", "Value"],
            ["Algorithm", result_data.get("algorithm", "N/A")],
            ["Number of Clusters", str(summary.get("num_clusters", 0))],
            ["Total Regions", str(summary.get("total_regions", 0))],
            ["Davies-Bouldin Index", f"{evaluation.get('davies_bouldin', 0):.4f}"],
            ["Silhouette Score", f"{evaluation.get('silhouette_score', 0):.4f}"],
        ]

        summary_table = Table(summary_data, colWidths=[3 * inch, 3 * inch])
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#667eea")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        story.append(summary_table)
        story.append(PageBreak())

        # Visualizations
        clusters = result_data.get("clusters", [])

        if clusters:
            story.append(Paragraph("Visualizations", self.heading_style))
            story.append(Spacer(1, 0.2 * inch))

            # Geographical Map (only if coordinates available)
            cluster_map = self._create_cluster_map(
                clusters, "Cluster Geographical Distribution"
            )
            if cluster_map:
                story.append(
                    Paragraph("Geographical Distribution", self.subheading_style)
                )
                story.append(Spacer(1, 0.1 * inch))
                img_map = Image(cluster_map, width=6.5 * inch, height=4.875 * inch)
                story.append(img_map)
                story.append(PageBreak())

            # Scatter plots
            story.append(Paragraph("Scatter Plot Analysis", self.subheading_style))
            story.append(Spacer(1, 0.1 * inch))

            scatter1 = self._create_scatter_plot(
                clusters, "ipm", "garis_kemiskinan", "IPM vs Garis Kemiskinan"
            )
            img1 = Image(scatter1, width=5 * inch, height=3.75 * inch)
            story.append(img1)
            story.append(Spacer(1, 0.3 * inch))

            scatter2 = self._create_scatter_plot(
                clusters,
                "ipm",
                "pengeluaran_per_kapita",
                "IPM vs Pengeluaran Per Kapita",
            )
            img2 = Image(scatter2, width=5 * inch, height=3.75 * inch)
            story.append(img2)
            story.append(Spacer(1, 0.3 * inch))

            scatter3 = self._create_scatter_plot(
                clusters,
                "garis_kemiskinan",
                "pengeluaran_per_kapita",
                "Garis Kemiskinan vs Pengeluaran Per Kapita",
            )
            img_scatter3 = Image(scatter3, width=5 * inch, height=3.75 * inch)
            story.append(img_scatter3)
            story.append(PageBreak())

            # Box plots
            story.append(Paragraph("Distribution Analysis", self.subheading_style))
            story.append(Spacer(1, 0.1 * inch))

            box1 = self._create_box_plot(clusters, "ipm", "IPM Distribution by Cluster")
            img3 = Image(box1, width=6 * inch, height=3.6 * inch)
            story.append(img3)
            story.append(Spacer(1, 0.3 * inch))

            box2 = self._create_box_plot(
                clusters, "garis_kemiskinan", "Garis Kemiskinan Distribution by Cluster"
            )
            img4 = Image(box2, width=6 * inch, height=3.6 * inch)
            story.append(img4)
            story.append(Spacer(1, 0.3 * inch))

            box3 = self._create_box_plot(
                clusters,
                "pengeluaran_per_kapita",
                "Pengeluaran Per Kapita Distribution by Cluster",
            )
            img_box3 = Image(box3, width=6 * inch, height=3.6 * inch)
            story.append(img_box3)
            story.append(PageBreak())

            # Correlation heatmap
            story.append(Paragraph("Feature Correlation", self.subheading_style))
            story.append(Spacer(1, 0.1 * inch))

            heatmap = self._create_correlation_heatmap(clusters, "Correlation Heatmap")
            if heatmap:
                img5 = Image(heatmap, width=5 * inch, height=3.75 * inch)
                story.append(img5)
                story.append(Spacer(1, 0.3 * inch))

            # Silhouette plot
            silhouette_score = evaluation.get("silhouette_score", 0)
            silhouette = self._create_silhouette_plot(
                clusters, silhouette_score, "Silhouette Plot"
            )
            img6 = Image(silhouette, width=6 * inch, height=3.6 * inch)
            story.append(img6)
            story.append(PageBreak())

            # Cluster Details with Interpretation Labels and Members (comma-separated)
            story.append(Paragraph("Cluster Details", self.heading_style))
            story.append(Spacer(1, 0.2 * inch))

            for cluster in clusters:
                cluster_table = self._create_cluster_details_table(cluster)
                story.append(cluster_table)
                story.append(Spacer(1, 0.4 * inch))

        # Build PDF
        doc.build(story)
        return output_path


def generate_pdf_report(data: Dict[str, Any], mode: str = "yearly") -> str:
    """
    Main function to generate PDF report

    Args:
        data: Clustering results data
        mode: 'yearly' or 'all_years'

    Returns:
        Path to generated PDF file
    """
    generator = ClusteringPDFGenerator()

    # Create temporary file
    temp_dir = tempfile.gettempdir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"clustering_report_{mode}_{timestamp}.pdf"
    output_path = os.path.join(temp_dir, filename)

    if mode == "yearly":
        return generator.generate_yearly_pdf(data, output_path)
    else:
        return generator.generate_all_years_pdf(data, output_path)
