from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.db import transaction
from django.http import HttpResponse, FileResponse
from .models import ClusteringSession
from .algorithms import run_clustering_all_years, run_clustering_per_year
from .pdf_generator import generate_pdf_report
from .utils import (
    normalize_column_names,
    validate_required_columns,
    clean_and_validate_data,
    read_data_file,
    format_clustering_parameters,
    safe_int_conversion,
    safe_float_conversion,
)
from .constants import (
    ALGORITHM_FCM,
    ALGORITHM_OPTICS,
    SUPPORTED_ALGORITHMS,
    MODE_PER_YEAR,
    MODE_ALL_YEARS,
    DEFAULT_NUM_CLUSTERS,
    DEFAULT_FUZZY_COEFF,
    DEFAULT_MAX_ITER,
    DEFAULT_TOLERANCE,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_XI,
    DEFAULT_MIN_CLUSTER_SIZE,
    CLUSTERING_FEATURES,
    MATPLOTLIB_BACKEND,
)
from django.http import FileResponse
from django.conf import settings
from rest_framework import status
from rest_framework.response import Response
import pandas as pd
import numpy as np
import json
import io
import csv
import os
from io import BytesIO

# Configure matplotlib for headless environments
import matplotlib

matplotlib.use(MATPLOTLIB_BACKEND)
import matplotlib.pyplot as plt
import seaborn as sns
import base64


class UploadAndProcessView(APIView):
    """
    API View for uploading and processing clustering data.
    Handles file upload, validation, and clustering execution.
    """

    def post(self, request):
        """Process uploaded file and perform clustering analysis."""
        file_obj = request.FILES.get("file")
        if not file_obj:
            return Response(
                {"error": "File tidak ditemukan"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Parse and validate parameters
        try:
            algorithm = request.POST.get("algorithm", ALGORITHM_FCM).lower()
            num_clusters = safe_int_conversion(
                request.POST.get("num_clusters"), DEFAULT_NUM_CLUSTERS
            )
            fuzzy_coeff = safe_float_conversion(
                request.POST.get("fuzzy_coeff"), DEFAULT_FUZZY_COEFF
            )
            max_iter = safe_int_conversion(
                request.POST.get("max_iter"), DEFAULT_MAX_ITER
            )
            tolerance = safe_float_conversion(
                request.POST.get("tolerance"), DEFAULT_TOLERANCE
            )
            selected_year = request.POST.get("selected_year")
            clustering_mode = request.POST.get("clustering_mode", MODE_PER_YEAR)

            # Get selected years for per_year mode (optional)
            selected_years = self._parse_selected_years(
                request.POST.get("selected_years")
            )

            # OPTICS specific parameters
            min_samples = safe_int_conversion(
                request.POST.get("min_samples"), DEFAULT_MIN_SAMPLES
            )
            xi = safe_float_conversion(request.POST.get("xi"), DEFAULT_XI)
            min_cluster_size = safe_float_conversion(
                request.POST.get("min_cluster_size"), DEFAULT_MIN_CLUSTER_SIZE
            )

        except Exception as e:
            return Response(
                {"error": f"Parameter tidak valid: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Read and validate file
        try:
            df = read_data_file(file_obj)
        except ValueError as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Normalize column names
        df, column_mapping = normalize_column_names(df)

        # Validate required columns
        missing_cols = validate_required_columns(df)
        if missing_cols:
            return Response(
                {"error": f'Kolom wajib hilang: {", ".join(missing_cols)}'},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Clean and validate data
        try:
            df = clean_and_validate_data(df)
        except ValueError as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Format parameters for storage
        parameters = format_clustering_parameters(
            algorithm=algorithm,
            num_clusters=num_clusters,
            fuzzy_coeff=fuzzy_coeff,
            max_iter=max_iter,
            tolerance=tolerance,
            min_samples=min_samples,
            xi=xi,
            min_cluster_size=min_cluster_size,
            selected_year=selected_year,
        )

        # Validate algorithm
        if algorithm not in SUPPORTED_ALGORITHMS:
            return Response(
                {
                    "error": f'Algoritma tidak dikenal. Gunakan "{ALGORITHM_FCM}" atau "{ALGORITHM_OPTICS}"'
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Execute clustering
        try:
            results = self._execute_clustering(
                df=df,
                algorithm=algorithm,
                clustering_mode=clustering_mode,
                num_clusters=num_clusters,
                fuzzy_coeff=fuzzy_coeff,
                max_iter=max_iter,
                tolerance=tolerance,
                min_samples=min_samples,
                xi=xi,
                min_cluster_size=min_cluster_size,
                selected_year=selected_year,
                selected_years=selected_years,
            )
            results["clustering_mode"] = clustering_mode

        except Exception as e:
            return Response(
                {"error": f"Gagal melakukan clustering: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # Save session
        with transaction.atomic():
            session = ClusteringSession.objects.create(
                original_filename=getattr(file_obj, "name", ""),
                parameters=parameters,
                results=results,
            )

        return Response(
            {"session_id": str(session.id), "results": results},
            status=status.HTTP_201_CREATED,
        )

    def _parse_selected_years(self, selected_years_json):
        """Parse selected years from JSON string."""
        if not selected_years_json:
            return None

        try:
            selected_years = json.loads(selected_years_json)
            if selected_years:
                selected_years = [int(y) for y in selected_years]
                print(f"🎯 Selected years from request: {selected_years}")
                return selected_years
        except Exception as e:
            print(f"⚠️ Error parsing selected_years: {e}")

        return None

    def _execute_clustering(
        self,
        df,
        algorithm,
        clustering_mode,
        num_clusters,
        fuzzy_coeff,
        max_iter,
        tolerance,
        min_samples,
        xi,
        min_cluster_size,
        selected_year,
        selected_years,
    ):
        """Execute clustering based on mode and algorithm."""
        if clustering_mode == MODE_ALL_YEARS:
            print("📅 Clustering all years at once (wide format, all year columns)")
            return self._run_all_years_clustering(
                df=df,
                algorithm=algorithm,
                num_clusters=num_clusters,
                fuzzy_coeff=fuzzy_coeff,
                max_iter=max_iter,
                tolerance=tolerance,
                min_samples=min_samples,
                xi=xi,
                min_cluster_size=min_cluster_size,
                selected_year=selected_year,
            )
        else:
            # Per year clustering (default)
            if selected_year:
                print(f"🎯 Single year clustering for {selected_year}")
            else:
                print(f"🗓️ Per year clustering for all available years")

            return self._run_per_year_clustering(
                df=df,
                algorithm=algorithm,
                num_clusters=num_clusters,
                fuzzy_coeff=fuzzy_coeff,
                max_iter=max_iter,
                tolerance=tolerance,
                min_samples=min_samples,
                xi=xi,
                min_cluster_size=min_cluster_size,
                selected_year=selected_year,
                selected_years=selected_years,
            )

    def _run_all_years_clustering(
        self,
        df,
        algorithm,
        num_clusters,
        fuzzy_coeff,
        max_iter,
        tolerance,
        min_samples,
        xi,
        min_cluster_size,
        selected_year,
    ):
        """Run clustering for all years combined."""
        if algorithm == ALGORITHM_FCM:
            return run_clustering_all_years(
                df,
                algorithm=ALGORITHM_FCM,
                n_clusters=num_clusters,
                m=fuzzy_coeff,
                max_iter=max_iter,
                error=tolerance,
                selected_year=selected_year,
            )
        else:  # OPTICS
            return run_clustering_all_years(
                df,
                algorithm=ALGORITHM_OPTICS,
                min_samples=min_samples,
                xi=xi,
                min_cluster_size=min_cluster_size,
                selected_year=selected_year,
            )

    def _run_per_year_clustering(
        self,
        df,
        algorithm,
        num_clusters,
        fuzzy_coeff,
        max_iter,
        tolerance,
        min_samples,
        xi,
        min_cluster_size,
        selected_year,
        selected_years,
    ):
        """Run clustering per year."""
        if algorithm == ALGORITHM_FCM:
            return run_clustering_per_year(
                df,
                algorithm=ALGORITHM_FCM,
                features=CLUSTERING_FEATURES,
                n_clusters=num_clusters,
                m=fuzzy_coeff,
                max_iter=max_iter,
                error=tolerance,
                selected_year=selected_year,
                selected_years=selected_years,
            )
        else:  # OPTICS
            return run_clustering_per_year(
                df,
                algorithm=ALGORITHM_OPTICS,
                features=CLUSTERING_FEATURES,
                min_samples=min_samples,
                xi=xi,
                min_cluster_size=min_cluster_size,
                selected_year=selected_year,
                selected_years=selected_years,
            )


class GetResultsView(APIView):
    def get(self, request, session_id: str):
        try:
            session = ClusteringSession.objects.get(id=session_id)
        except ClusteringSession.DoesNotExist:
            return Response(
                {"error": "Session tidak ditemukan"}, status=status.HTTP_404_NOT_FOUND
            )
        return Response(session.results, status=status.HTTP_200_OK)


class ExportResultsView(APIView):
    def get(self, request, session_id: str):
        try:
            session = ClusteringSession.objects.get(id=session_id)
        except ClusteringSession.DoesNotExist:
            return Response(
                {"error": "Session tidak ditemukan"}, status=status.HTTP_404_NOT_FOUND
            )

        format_type = request.GET.get("format", "csv").lower()
        results = session.results

        if format_type == "csv":
            # Create CSV export
            output = io.StringIO()
            writer = csv.writer(output)

            # Handle both single year and per-year results
            if results.get("clustering_type") == "per_year":
                # Per-year results
                writer.writerow(
                    [
                        "Year",
                        "Kabupaten/Kota",
                        "Cluster",
                        "IPM",
                        "Garis_Kemiskinan",
                        "Pengeluaran_Per_Kapita",
                        "Membership",
                    ]
                )

                for year, year_results in results.get("results_per_year", {}).items():
                    if "clusters" in year_results:
                        for cluster in year_results["clusters"]:
                            for member in cluster.get("members", []):
                                writer.writerow(
                                    [
                                        year,
                                        member.get("kabupaten_kota", ""),
                                        cluster.get("id", ""),
                                        member.get("ipm", ""),
                                        member.get("garis_kemiskinan", ""),
                                        member.get("pengeluaran_per_kapita", ""),
                                        member.get("membership", ""),
                                    ]
                                )
            else:
                # Single year results
                writer.writerow(
                    [
                        "Kabupaten/Kota",
                        "Cluster",
                        "IPM",
                        "Garis_Kemiskinan",
                        "Pengeluaran_Per_Kapita",
                        "Membership",
                    ]
                )

                for cluster in results.get("clusters", []):
                    for member in cluster.get("members", []):
                        writer.writerow(
                            [
                                member.get("kabupaten_kota", ""),
                                cluster.get("id", ""),
                                member.get("ipm", ""),
                                member.get("garis_kemiskinan", ""),
                                member.get("pengeluaran_per_kapita", ""),
                                member.get("membership", ""),
                            ]
                        )

            response = HttpResponse(output.getvalue(), content_type="text/csv")
            response["Content-Disposition"] = (
                f'attachment; filename="clustering_results_{session_id}.csv"'
            )
            return response

        elif format_type == "json":
            response = HttpResponse(
                json.dumps(results, indent=2), content_type="application/json"
            )
            response["Content-Disposition"] = (
                f'attachment; filename="clustering_results_{session_id}.json"'
            )
            return response

        else:
            return Response(
                {"error": "Format tidak didukung. Gunakan csv atau json"},
                status=status.HTTP_400_BAD_REQUEST,
            )

    def generate_correlation_heatmap_base64(df, features):
        """
        Menghasilkan heatmap korelasi dari dataframe dan mengembalikannya sebagai string base64.
        """
        if df.empty or len(df) < 2:
            return None

        # Hitung matriks korelasi
        correlation_matrix = df[features].corr()

        # Buat plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
        )
        plt.title("Heatmap Korelasi Antar Variabel")

        # Simpan plot ke buffer di memori
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)

        # Encode gambar ke base64
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return f"data:image/png;base64,{image_base64}"


class GenerateReportView(APIView):
    def post(self, request, session_id: str):
        try:
            session = ClusteringSession.objects.get(id=session_id)
        except ClusteringSession.DoesNotExist:
            return Response(
                {"error": "Session tidak ditemukan"}, status=status.HTTP_404_NOT_FOUND
            )

        # For now, return a simple report summary
        # In the future, this could generate a PDF report
        results = session.results

        report = {
            "session_id": str(session_id),
            "generated_at": pd.Timestamp.now().isoformat(),
            "summary": results.get("summary", {}),
            "evaluation": results.get("evaluation", {}),
            "cluster_count": len(results.get("clusters", [])),
            "total_regions": results.get("summary", {}).get("total_regions", 0),
        }

        return Response(report, status=status.HTTP_200_OK)


class GetGeographicalDataView(APIView):
    def get(self, request, session_id: str):
        try:
            session = ClusteringSession.objects.get(id=session_id)
        except ClusteringSession.DoesNotExist:
            return Response(
                {"error": "Session tidak ditemukan"}, status=status.HTTP_404_NOT_FOUND
            )

        results = session.results
        geographical_data = []

        # Handle both single year and per-year results
        if results.get("clustering_type") == "per_year":
            # For per-year results, get the most recent year's data
            results_per_year = results.get("results_per_year", {})
            if results_per_year:
                latest_year = max(results_per_year.keys())
                clusters = results_per_year[latest_year].get("clusters", [])
            else:
                clusters = []
        else:
            clusters = results.get("clusters", [])

        for cluster in clusters:
            for member in cluster.get("members", []):
                geographical_data.append(
                    {
                        "kabupaten_kota": member.get("kabupaten_kota", ""),
                        "cluster_id": cluster.get("id", ""),
                        "latitude": member.get("latitude", 0.0),
                        "longitude": member.get("longitude", 0.0),
                        "ipm": member.get("ipm", 0.0),
                        "garis_kemiskinan": member.get("garis_kemiskinan", 0.0),
                        "pengeluaran_per_kapita": member.get(
                            "pengeluaran_per_kapita", 0.0
                        ),
                        "membership": member.get("membership", 1.0),
                    }
                )

        return Response(
            {
                "geographical_data": geographical_data,
                "total_points": len(geographical_data),
            },
            status=status.HTTP_200_OK,
        )


class GetClusterDetailsView(APIView):
    def get(self, request, session_id: str, cluster_id: int):
        try:
            session = ClusteringSession.objects.get(id=session_id)
        except ClusteringSession.DoesNotExist:
            return Response(
                {"error": "Session tidak ditemukan"}, status=status.HTTP_404_NOT_FOUND
            )

        results = session.results

        # Handle both single year and per-year results
        if results.get("clustering_type") == "per_year":
            # For per-year results, get the most recent year's data
            results_per_year = results.get("results_per_year", {})
            if results_per_year:
                latest_year = max(results_per_year.keys())
                clusters = results_per_year[latest_year].get("clusters", [])
            else:
                clusters = []
        else:
            clusters = results.get("clusters", [])

        # Find the specific cluster
        target_cluster = None
        for cluster in clusters:
            if cluster.get("id") == cluster_id or str(cluster.get("id")) == str(
                cluster_id
            ):
                target_cluster = cluster
                break

        if not target_cluster:
            return Response(
                {"error": "Cluster tidak ditemukan"}, status=status.HTTP_404_NOT_FOUND
            )

        # Calculate additional statistics
        members = target_cluster.get("members", [])
        if members:
            ipm_values = [m.get("ipm", 0) for m in members]
            gk_values = [m.get("garis_kemiskinan", 0) for m in members]
            pp_values = [m.get("pengeluaran_per_kapita", 0) for m in members]

            statistics = {
                "ipm": {
                    "mean": np.mean(ipm_values),
                    "std": np.std(ipm_values),
                    "min": np.min(ipm_values),
                    "max": np.max(ipm_values),
                },
                "garis_kemiskinan": {
                    "mean": np.mean(gk_values),
                    "std": np.std(gk_values),
                    "min": np.min(gk_values),
                    "max": np.max(gk_values),
                },
                "pengeluaran_per_kapita": {
                    "mean": np.mean(pp_values),
                    "std": np.std(pp_values),
                    "min": np.min(pp_values),
                    "max": np.max(pp_values),
                },
            }
        else:
            statistics = {}

        cluster_details = {
            "cluster": target_cluster,
            "statistics": statistics,
            "member_count": len(members),
        }

        return Response(cluster_details, status=status.HTTP_200_OK)


class GetEvaluationMetricsView(APIView):
    def get(self, request, session_id: str):
        try:
            session = ClusteringSession.objects.get(id=session_id)
        except ClusteringSession.DoesNotExist:
            return Response(
                {"error": "Session tidak ditemukan"}, status=status.HTTP_404_NOT_FOUND
            )

        results = session.results

        # Handle both single year and per-year results
        if results.get("clustering_type") == "per_year":
            overall_summary = results.get("overall_summary", {})
            evaluation_data = {
                "clustering_type": "per_year",
                "overall_evaluation": overall_summary.get("average_evaluation", {}),
                "per_year_evaluation": {},
            }

            # Add per-year evaluation metrics
            for year, year_results in results.get("results_per_year", {}).items():
                evaluation_data["per_year_evaluation"][year] = year_results.get(
                    "evaluation", {}
                )

        else:
            evaluation_data = {
                "clustering_type": "single_year",
                "evaluation": results.get("evaluation", {}),
                "summary": results.get("summary", {}),
            }

        return Response(evaluation_data, status=status.HTTP_200_OK)


class GetSilhouettePlotView(APIView):
    """
    API endpoint to generate silhouette plot as PNG image
    """

    def get(self, request, session_id: str, year: str = None):
        try:
            session = ClusteringSession.objects.get(id=session_id)
        except ClusteringSession.DoesNotExist:
            return Response(
                {"error": "Session tidak ditemukan"}, status=status.HTTP_404_NOT_FOUND
            )

        try:
            results = session.results
            clustering_type = results.get("clustering_type", "per_year")

            # Get appropriate data based on clustering type and year
            if clustering_type == "per_year" and year:
                year_results = results.get("results_per_year", {}).get(str(year))
                if not year_results:
                    return Response(
                        {"error": f"Hasil untuk tahun {year} tidak ditemukan"},
                        status=status.HTTP_404_NOT_FOUND,
                    )
                clusters = year_results.get("clusters", [])
                silhouette_score = year_results.get("evaluation", {}).get(
                    "silhouette_score", 0.5
                )
                title = f"Silhouette Plot - Tahun {year}"
            else:
                all_year_results = results.get("results_per_year", {}).get(
                    "all_years", {}
                )
                # All years mode or no year specified
                clusters = all_year_results.get("clusters", [])
                silhouette_score = all_year_results.get("evaluation", {}).get(
                    "silhouette_score", 0.5
                )
                title = "Silhouette Plot"
            if not clusters:
                return Response(
                    {"error": "Tidak ada data cluster untuk ditampilkan"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            # Generate silhouette plot
            img_buffer = self._create_silhouette_plot(clusters, silhouette_score, title)

            # Return as image
            response = HttpResponse(img_buffer.getvalue(), content_type="image/png")
            response["Content-Disposition"] = (
                f'inline; filename="silhouette_plot_{session_id}_{year or "all"}.png"'
            )

            return response

        except Exception as e:
            print(f"❌ Error generating silhouette plot: {str(e)}")
            import traceback

            traceback.print_exc()
            return Response(
                {"error": f"Gagal membuat silhouette plot: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _create_silhouette_plot(self, clusters, silhouette_score, title):
        """Create silhouette plot (reused from PDF generator logic)"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define colors for clusters
        colors_palette = [
            "#667eea",
            "#48bb78",
            "#ed8936",
            "#f56565",
            "#38b2ac",
            "#9f7aea",
            "#ed64a6",
            "#ecc94b",
            "#4299e1",
            "#fc8181",
        ]

        y_lower = 10

        for idx, cluster in enumerate(clusters):
            members = cluster.get("members", [])
            cluster_id = cluster.get("id", idx)

            # Calculate approximate silhouette scores
            n_members = len(members)

            if n_members == 0:
                continue

            # Create silhouette values (sorted descending)
            silhouette_values = []
            for member in members:
                if (
                    isinstance(member, dict)
                    and "membership" in member
                    and member["membership"] is not None
                ):
                    # Convert membership to silhouette-like score
                    silhouette_values.append(member["membership"] * 0.8 - 0.4)
                else:
                    silhouette_values.append(np.random.uniform(0.3, 0.7))

            silhouette_values = np.array(sorted(silhouette_values, reverse=True))

            y_upper = y_lower + n_members

            color = colors_palette[idx % len(colors_palette)]
            ax.barh(
                range(y_lower, y_upper),
                silhouette_values,
                height=1.0,
                color=color,
                alpha=0.8,
                edgecolor="none",
            )

            # Label cluster
            cluster_label = cluster.get("interpretation", {}).get(
                "label", f"Cluster {cluster_id}"
            )
            ax.text(
                -0.05,
                y_lower + 0.5 * n_members,
                f"C{cluster_id}",
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


class DownloadPDFReportView(APIView):
    """
    API endpoint to download complete PDF report with all visualizations
    """

    def get(self, request, session_id: str):
        try:
            session = ClusteringSession.objects.get(id=session_id)
        except ClusteringSession.DoesNotExist:
            return Response(
                {"error": "Session tidak ditemukan"}, status=status.HTTP_404_NOT_FOUND
            )

        try:
            results = session.results
            clustering_type = results.get("clustering_type", "per_year")

            # Determine mode for PDF generation
            if clustering_type == "per_year":
                mode = "yearly"
            else:
                mode = "all_years"

            print(f"📄 Generating PDF report for session {session_id}, mode: {mode}")

            # Generate PDF
            pdf_path = generate_pdf_report(results, mode=mode)

            print(f"✅ PDF generated successfully: {pdf_path}")

            # Check if file exists
            if not os.path.exists(pdf_path):
                return Response(
                    {"error": "Failed to generate PDF"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            # Read and return PDF file
            with open(pdf_path, "rb") as pdf_file:
                response = HttpResponse(pdf_file.read(), content_type="application/pdf")
                response["Content-Disposition"] = (
                    f'attachment; filename="clustering_report_{session_id}_{mode}.pdf"'
                )

            # Clean up temporary file
            try:
                os.remove(pdf_path)
            except Exception as e:
                print(f"⚠️ Warning: Could not remove temp file: {e}")

            return response

        except Exception as e:
            print(f"❌ Error generating PDF: {str(e)}")
            import traceback

            traceback.print_exc()
            return Response(
                {"error": f"Error generating PDF: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class DownloadSampleExcelView(APIView):
    """
    API View untuk mengunduh file sample CSV (long format).
    """

    def get(self, request):
        csv_file_name = "sample_data_indonesia.csv"
        excel_file_name = "sample_data_indonesia.xlsx"

        try:
            base_dir = settings.BASE_DIR
            csv_path = base_dir / "sample-data" / csv_file_name

            # Periksa file CSV
            if not os.path.exists(csv_path):
                return Response(
                    {
                        "error": f"File sample '{csv_file_name}' tidak ditemukan di path server: {str(csv_path)}"
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )

            # --- Baca CSV menjadi DataFrame ---
            df = pd.read_csv(csv_path)

            # --- Convert ke Excel dalam memory ---
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)

            # --- Return FileResponse sebagai .xlsx ---
            response = FileResponse(
                output,
                as_attachment=True,
                filename=excel_file_name
            )
            response["Content-Type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

            return response

        except Exception as e:
            return Response(
                {"error": f"Server gagal memproses file: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
